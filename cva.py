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




__kernel void builtinzhreplicate_f32zireplicate_17892(__global
                                                      unsigned char *mem_17888,
                                                      int32_t num_elems_17889,
                                                      float val_17890)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_17892;
    int32_t replicate_ltid_17893;
    int32_t replicate_gid_17894;
    
    replicate_gtid_17892 = get_global_id(0);
    replicate_ltid_17893 = get_local_id(0);
    replicate_gid_17894 = get_group_id(0);
    if (slt64(replicate_gtid_17892, num_elems_17889)) {
        ((__global float *) mem_17888)[sext_i32_i64(replicate_gtid_17892)] =
            val_17890;
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_i64zireplicate_17484(__global
                                                      unsigned char *mem_17480,
                                                      int32_t num_elems_17481,
                                                      int64_t val_17482)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_17484;
    int32_t replicate_ltid_17485;
    int32_t replicate_gid_17486;
    
    replicate_gtid_17484 = get_global_id(0);
    replicate_ltid_17485 = get_local_id(0);
    replicate_gid_17486 = get_group_id(0);
    if (slt64(replicate_gtid_17484, num_elems_17481)) {
        ((__global int64_t *) mem_17480)[sext_i32_i64(replicate_gtid_17484)] =
            val_17482;
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
__kernel void mainziscan_stage1_17172(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17438_backing_aligned_0,
                                      int64_t paths_16408, int64_t res_16516,
                                      __global unsigned char *mem_17348,
                                      int32_t num_threads_17432)
{
    #define segscan_group_sizze_17167 (mainzisegscan_group_sizze_17166)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17438_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17438_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17433;
    int32_t local_tid_17434;
    int64_t group_sizze_17437;
    int32_t wave_sizze_17436;
    int32_t group_tid_17435;
    
    global_tid_17433 = get_global_id(0);
    local_tid_17434 = get_local_id(0);
    group_sizze_17437 = get_local_size(0);
    wave_sizze_17436 = LOCKSTEP_WIDTH;
    group_tid_17435 = get_group_id(0);
    
    int32_t phys_tid_17172;
    
    phys_tid_17172 = global_tid_17433;
    
    __local char *scan_arr_mem_17438;
    
    scan_arr_mem_17438 = (__local char *) scan_arr_mem_17438_backing_0;
    
    int64_t x_16519;
    int64_t x_16520;
    
    x_16519 = 0;
    for (int64_t j_17440 = 0; j_17440 < sdiv_up64(paths_16408,
                                                  sext_i32_i64(num_threads_17432));
         j_17440++) {
        int64_t chunk_offset_17441 = segscan_group_sizze_17167 * j_17440 +
                sext_i32_i64(group_tid_17435) * (segscan_group_sizze_17167 *
                                                 sdiv_up64(paths_16408,
                                                           sext_i32_i64(num_threads_17432)));
        int64_t flat_idx_17442 = chunk_offset_17441 +
                sext_i32_i64(local_tid_17434);
        int64_t gtid_17171 = flat_idx_17442;
        
        // threads in bounds read input
        {
            if (slt64(gtid_17171, paths_16408)) {
                // write to-scan values to parameters
                {
                    x_16520 = res_16516;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_17171, paths_16408)) {
                    x_16520 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int64_t res_16521 = add64(x_16519, x_16520);
                
                ((__local
                  int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)] =
                    res_16521;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int64_t x_17443;
            int64_t x_17444;
            int64_t x_17446;
            int64_t x_17447;
            bool ltid_in_bounds_17449;
            
            ltid_in_bounds_17449 = slt64(sext_i32_i64(local_tid_17434),
                                         segscan_group_sizze_17167);
            
            int32_t skip_threads_17450;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_17449) {
                    x_17444 = ((volatile __local
                                int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)];
                    if ((local_tid_17434 - squot32(local_tid_17434, 32) * 32) ==
                        0) {
                        x_17443 = x_17444;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17450 = 1;
                while (slt32(skip_threads_17450, 32)) {
                    if (sle32(skip_threads_17450, local_tid_17434 -
                              squot32(local_tid_17434, 32) * 32) &&
                        ltid_in_bounds_17449) {
                        // read operands
                        {
                            x_17443 = ((volatile __local
                                        int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434) -
                                                                       sext_i32_i64(skip_threads_17450)];
                        }
                        // perform operation
                        {
                            int64_t res_17445 = add64(x_17443, x_17444);
                            
                            x_17443 = res_17445;
                        }
                    }
                    if (sle32(wave_sizze_17436, skip_threads_17450)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17450, local_tid_17434 -
                              squot32(local_tid_17434, 32) * 32) &&
                        ltid_in_bounds_17449) {
                        // write result
                        {
                            ((volatile __local
                              int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)] =
                                x_17443;
                            x_17444 = x_17443;
                        }
                    }
                    if (sle32(wave_sizze_17436, skip_threads_17450)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17450 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17434 - squot32(local_tid_17434, 32) * 32) ==
                    31 && ltid_in_bounds_17449) {
                    ((volatile __local
                      int64_t *) scan_arr_mem_17438)[sext_i32_i64(squot32(local_tid_17434,
                                                                          32))] =
                        x_17443;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17451;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17434, 32) == 0 &&
                        ltid_in_bounds_17449) {
                        x_17447 = ((volatile __local
                                    int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)];
                        if ((local_tid_17434 - squot32(local_tid_17434, 32) *
                             32) == 0) {
                            x_17446 = x_17447;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17451 = 1;
                    while (slt32(skip_threads_17451, 32)) {
                        if (sle32(skip_threads_17451, local_tid_17434 -
                                  squot32(local_tid_17434, 32) * 32) &&
                            (squot32(local_tid_17434, 32) == 0 &&
                             ltid_in_bounds_17449)) {
                            // read operands
                            {
                                x_17446 = ((volatile __local
                                            int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434) -
                                                                           sext_i32_i64(skip_threads_17451)];
                            }
                            // perform operation
                            {
                                int64_t res_17448 = add64(x_17446, x_17447);
                                
                                x_17446 = res_17448;
                            }
                        }
                        if (sle32(wave_sizze_17436, skip_threads_17451)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17451, local_tid_17434 -
                                  squot32(local_tid_17434, 32) * 32) &&
                            (squot32(local_tid_17434, 32) == 0 &&
                             ltid_in_bounds_17449)) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)] =
                                    x_17446;
                                x_17447 = x_17446;
                            }
                        }
                        if (sle32(wave_sizze_17436, skip_threads_17451)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17451 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17434, 32) == 0 ||
                      !ltid_in_bounds_17449)) {
                    // read operands
                    {
                        x_17444 = x_17443;
                        x_17443 = ((__local
                                    int64_t *) scan_arr_mem_17438)[sext_i32_i64(squot32(local_tid_17434,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        int64_t res_17445 = add64(x_17443, x_17444);
                        
                        x_17443 = res_17445;
                    }
                    // write final result
                    {
                        ((__local
                          int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)] =
                            x_17443;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17434, 32) == 0) {
                    ((__local
                      int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)] =
                        x_17444;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_17171, paths_16408)) {
                    ((__global int64_t *) mem_17348)[gtid_17171] = ((__local
                                                                     int64_t *) scan_arr_mem_17438)[sext_i32_i64(local_tid_17434)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17452 = 0;
                bool should_load_carry_17453 = local_tid_17434 == 0 &&
                     !crosses_segment_17452;
                
                if (should_load_carry_17453) {
                    x_16519 = ((__local
                                int64_t *) scan_arr_mem_17438)[segscan_group_sizze_17167 -
                                                               1];
                }
                if (!should_load_carry_17453) {
                    x_16519 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_17167
}
__kernel void mainziscan_stage1_17195(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17647_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_17645_backing_aligned_1,
                                      int64_t res_16524, __global
                                      unsigned char *mem_17350, __global
                                      unsigned char *mem_17354, __global
                                      unsigned char *mem_17356,
                                      int32_t num_threads_17639)
{
    #define segscan_group_sizze_17190 (mainzisegscan_group_sizze_17189)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17647_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17647_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17645_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17645_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17640;
    int32_t local_tid_17641;
    int64_t group_sizze_17644;
    int32_t wave_sizze_17643;
    int32_t group_tid_17642;
    
    global_tid_17640 = get_global_id(0);
    local_tid_17641 = get_local_id(0);
    group_sizze_17644 = get_local_size(0);
    wave_sizze_17643 = LOCKSTEP_WIDTH;
    group_tid_17642 = get_group_id(0);
    
    int32_t phys_tid_17195;
    
    phys_tid_17195 = global_tid_17640;
    
    __local char *scan_arr_mem_17645;
    __local char *scan_arr_mem_17647;
    
    scan_arr_mem_17645 = (__local char *) scan_arr_mem_17645_backing_0;
    scan_arr_mem_17647 = (__local char *) scan_arr_mem_17647_backing_1;
    
    bool x_16540;
    int64_t x_16541;
    bool x_16542;
    int64_t x_16543;
    
    x_16540 = 0;
    x_16541 = 0;
    for (int64_t j_17649 = 0; j_17649 < sdiv_up64(res_16524,
                                                  sext_i32_i64(num_threads_17639));
         j_17649++) {
        int64_t chunk_offset_17650 = segscan_group_sizze_17190 * j_17649 +
                sext_i32_i64(group_tid_17642) * (segscan_group_sizze_17190 *
                                                 sdiv_up64(res_16524,
                                                           sext_i32_i64(num_threads_17639)));
        int64_t flat_idx_17651 = chunk_offset_17650 +
                sext_i32_i64(local_tid_17641);
        int64_t gtid_17194 = flat_idx_17651;
        
        // threads in bounds read input
        {
            if (slt64(gtid_17194, res_16524)) {
                int64_t x_16547 = ((__global int64_t *) mem_17350)[gtid_17194];
                bool res_16548 = slt64(0, x_16547);
                
                // write to-scan values to parameters
                {
                    x_16542 = res_16548;
                    x_16543 = x_16547;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_17194, res_16524)) {
                    x_16542 = 0;
                    x_16543 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_16544 = x_16540 || x_16542;
                int64_t res_16545;
                
                if (x_16542) {
                    res_16545 = x_16543;
                } else {
                    int64_t res_16546 = add64(x_16541, x_16543);
                    
                    res_16545 = res_16546;
                }
                ((__local
                  bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)] =
                    res_16544;
                ((__local
                  int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)] =
                    res_16545;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_17652;
            int64_t x_17653;
            bool x_17654;
            int64_t x_17655;
            bool x_17659;
            int64_t x_17660;
            bool x_17661;
            int64_t x_17662;
            bool ltid_in_bounds_17666;
            
            ltid_in_bounds_17666 = slt64(sext_i32_i64(local_tid_17641),
                                         segscan_group_sizze_17190);
            
            int32_t skip_threads_17667;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_17666) {
                    x_17654 = ((volatile __local
                                bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)];
                    x_17655 = ((volatile __local
                                int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)];
                    if ((local_tid_17641 - squot32(local_tid_17641, 32) * 32) ==
                        0) {
                        x_17652 = x_17654;
                        x_17653 = x_17655;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17667 = 1;
                while (slt32(skip_threads_17667, 32)) {
                    if (sle32(skip_threads_17667, local_tid_17641 -
                              squot32(local_tid_17641, 32) * 32) &&
                        ltid_in_bounds_17666) {
                        // read operands
                        {
                            x_17652 = ((volatile __local
                                        bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641) -
                                                                    sext_i32_i64(skip_threads_17667)];
                            x_17653 = ((volatile __local
                                        int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641) -
                                                                       sext_i32_i64(skip_threads_17667)];
                        }
                        // perform operation
                        {
                            bool res_17656 = x_17652 || x_17654;
                            int64_t res_17657;
                            
                            if (x_17654) {
                                res_17657 = x_17655;
                            } else {
                                int64_t res_17658 = add64(x_17653, x_17655);
                                
                                res_17657 = res_17658;
                            }
                            x_17652 = res_17656;
                            x_17653 = res_17657;
                        }
                    }
                    if (sle32(wave_sizze_17643, skip_threads_17667)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17667, local_tid_17641 -
                              squot32(local_tid_17641, 32) * 32) &&
                        ltid_in_bounds_17666) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)] =
                                x_17652;
                            x_17654 = x_17652;
                            ((volatile __local
                              int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)] =
                                x_17653;
                            x_17655 = x_17653;
                        }
                    }
                    if (sle32(wave_sizze_17643, skip_threads_17667)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17667 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17641 - squot32(local_tid_17641, 32) * 32) ==
                    31 && ltid_in_bounds_17666) {
                    ((volatile __local
                      bool *) scan_arr_mem_17645)[sext_i32_i64(squot32(local_tid_17641,
                                                                       32))] =
                        x_17652;
                    ((volatile __local
                      int64_t *) scan_arr_mem_17647)[sext_i32_i64(squot32(local_tid_17641,
                                                                          32))] =
                        x_17653;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17668;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17641, 32) == 0 &&
                        ltid_in_bounds_17666) {
                        x_17661 = ((volatile __local
                                    bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)];
                        x_17662 = ((volatile __local
                                    int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)];
                        if ((local_tid_17641 - squot32(local_tid_17641, 32) *
                             32) == 0) {
                            x_17659 = x_17661;
                            x_17660 = x_17662;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17668 = 1;
                    while (slt32(skip_threads_17668, 32)) {
                        if (sle32(skip_threads_17668, local_tid_17641 -
                                  squot32(local_tid_17641, 32) * 32) &&
                            (squot32(local_tid_17641, 32) == 0 &&
                             ltid_in_bounds_17666)) {
                            // read operands
                            {
                                x_17659 = ((volatile __local
                                            bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641) -
                                                                        sext_i32_i64(skip_threads_17668)];
                                x_17660 = ((volatile __local
                                            int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641) -
                                                                           sext_i32_i64(skip_threads_17668)];
                            }
                            // perform operation
                            {
                                bool res_17663 = x_17659 || x_17661;
                                int64_t res_17664;
                                
                                if (x_17661) {
                                    res_17664 = x_17662;
                                } else {
                                    int64_t res_17665 = add64(x_17660, x_17662);
                                    
                                    res_17664 = res_17665;
                                }
                                x_17659 = res_17663;
                                x_17660 = res_17664;
                            }
                        }
                        if (sle32(wave_sizze_17643, skip_threads_17668)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17668, local_tid_17641 -
                                  squot32(local_tid_17641, 32) * 32) &&
                            (squot32(local_tid_17641, 32) == 0 &&
                             ltid_in_bounds_17666)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)] =
                                    x_17659;
                                x_17661 = x_17659;
                                ((volatile __local
                                  int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)] =
                                    x_17660;
                                x_17662 = x_17660;
                            }
                        }
                        if (sle32(wave_sizze_17643, skip_threads_17668)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17668 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17641, 32) == 0 ||
                      !ltid_in_bounds_17666)) {
                    // read operands
                    {
                        x_17654 = x_17652;
                        x_17655 = x_17653;
                        x_17652 = ((__local
                                    bool *) scan_arr_mem_17645)[sext_i32_i64(squot32(local_tid_17641,
                                                                                     32)) -
                                                                1];
                        x_17653 = ((__local
                                    int64_t *) scan_arr_mem_17647)[sext_i32_i64(squot32(local_tid_17641,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        bool res_17656 = x_17652 || x_17654;
                        int64_t res_17657;
                        
                        if (x_17654) {
                            res_17657 = x_17655;
                        } else {
                            int64_t res_17658 = add64(x_17653, x_17655);
                            
                            res_17657 = res_17658;
                        }
                        x_17652 = res_17656;
                        x_17653 = res_17657;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)] =
                            x_17652;
                        ((__local
                          int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)] =
                            x_17653;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17641, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)] =
                        x_17654;
                    ((__local
                      int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)] =
                        x_17655;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_17194, res_16524)) {
                    ((__global bool *) mem_17354)[gtid_17194] = ((__local
                                                                  bool *) scan_arr_mem_17645)[sext_i32_i64(local_tid_17641)];
                    ((__global int64_t *) mem_17356)[gtid_17194] = ((__local
                                                                     int64_t *) scan_arr_mem_17647)[sext_i32_i64(local_tid_17641)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17669 = 0;
                bool should_load_carry_17670 = local_tid_17641 == 0 &&
                     !crosses_segment_17669;
                
                if (should_load_carry_17670) {
                    x_16540 = ((__local
                                bool *) scan_arr_mem_17645)[segscan_group_sizze_17190 -
                                                            1];
                    x_16541 = ((__local
                                int64_t *) scan_arr_mem_17647)[segscan_group_sizze_17190 -
                                                               1];
                }
                if (!should_load_carry_17670) {
                    x_16540 = 0;
                    x_16541 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_17190
}
__kernel void mainziscan_stage1_17203(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17714_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_17712_backing_aligned_1,
                                      int64_t res_16524, __global
                                      unsigned char *mem_17356, __global
                                      unsigned char *mem_17359, __global
                                      unsigned char *mem_17361, __global
                                      unsigned char *mem_17363,
                                      int32_t num_threads_17706)
{
    #define segscan_group_sizze_17198 (mainzisegscan_group_sizze_17197)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17714_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17714_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17712_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17712_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17707;
    int32_t local_tid_17708;
    int64_t group_sizze_17711;
    int32_t wave_sizze_17710;
    int32_t group_tid_17709;
    
    global_tid_17707 = get_global_id(0);
    local_tid_17708 = get_local_id(0);
    group_sizze_17711 = get_local_size(0);
    wave_sizze_17710 = LOCKSTEP_WIDTH;
    group_tid_17709 = get_group_id(0);
    
    int32_t phys_tid_17203;
    
    phys_tid_17203 = global_tid_17707;
    
    __local char *scan_arr_mem_17712;
    __local char *scan_arr_mem_17714;
    
    scan_arr_mem_17712 = (__local char *) scan_arr_mem_17712_backing_0;
    scan_arr_mem_17714 = (__local char *) scan_arr_mem_17714_backing_1;
    
    bool x_16579;
    int64_t x_16580;
    bool x_16581;
    int64_t x_16582;
    
    x_16579 = 0;
    x_16580 = 0;
    for (int64_t j_17716 = 0; j_17716 < sdiv_up64(res_16524,
                                                  sext_i32_i64(num_threads_17706));
         j_17716++) {
        int64_t chunk_offset_17717 = segscan_group_sizze_17198 * j_17716 +
                sext_i32_i64(group_tid_17709) * (segscan_group_sizze_17198 *
                                                 sdiv_up64(res_16524,
                                                           sext_i32_i64(num_threads_17706)));
        int64_t flat_idx_17718 = chunk_offset_17717 +
                sext_i32_i64(local_tid_17708);
        int64_t gtid_17202 = flat_idx_17718;
        
        // threads in bounds read input
        {
            if (slt64(gtid_17202, res_16524)) {
                int64_t x_16586 = ((__global int64_t *) mem_17356)[gtid_17202];
                int64_t i_p_o_17289 = add64(-1, gtid_17202);
                int64_t rot_i_17290 = smod64(i_p_o_17289, res_16524);
                int64_t x_16587 = ((__global int64_t *) mem_17356)[rot_i_17290];
                bool res_16589 = x_16586 == x_16587;
                bool res_16590 = !res_16589;
                
                // write to-scan values to parameters
                {
                    x_16581 = res_16590;
                    x_16582 = 1;
                }
                // write mapped values results to global memory
                {
                    ((__global bool *) mem_17363)[gtid_17202] = res_16590;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_17202, res_16524)) {
                    x_16581 = 0;
                    x_16582 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_16583 = x_16579 || x_16581;
                int64_t res_16584;
                
                if (x_16581) {
                    res_16584 = x_16582;
                } else {
                    int64_t res_16585 = add64(x_16580, x_16582);
                    
                    res_16584 = res_16585;
                }
                ((__local
                  bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)] =
                    res_16583;
                ((__local
                  int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)] =
                    res_16584;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_17719;
            int64_t x_17720;
            bool x_17721;
            int64_t x_17722;
            bool x_17726;
            int64_t x_17727;
            bool x_17728;
            int64_t x_17729;
            bool ltid_in_bounds_17733;
            
            ltid_in_bounds_17733 = slt64(sext_i32_i64(local_tid_17708),
                                         segscan_group_sizze_17198);
            
            int32_t skip_threads_17734;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_17733) {
                    x_17721 = ((volatile __local
                                bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)];
                    x_17722 = ((volatile __local
                                int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)];
                    if ((local_tid_17708 - squot32(local_tid_17708, 32) * 32) ==
                        0) {
                        x_17719 = x_17721;
                        x_17720 = x_17722;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17734 = 1;
                while (slt32(skip_threads_17734, 32)) {
                    if (sle32(skip_threads_17734, local_tid_17708 -
                              squot32(local_tid_17708, 32) * 32) &&
                        ltid_in_bounds_17733) {
                        // read operands
                        {
                            x_17719 = ((volatile __local
                                        bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708) -
                                                                    sext_i32_i64(skip_threads_17734)];
                            x_17720 = ((volatile __local
                                        int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708) -
                                                                       sext_i32_i64(skip_threads_17734)];
                        }
                        // perform operation
                        {
                            bool res_17723 = x_17719 || x_17721;
                            int64_t res_17724;
                            
                            if (x_17721) {
                                res_17724 = x_17722;
                            } else {
                                int64_t res_17725 = add64(x_17720, x_17722);
                                
                                res_17724 = res_17725;
                            }
                            x_17719 = res_17723;
                            x_17720 = res_17724;
                        }
                    }
                    if (sle32(wave_sizze_17710, skip_threads_17734)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17734, local_tid_17708 -
                              squot32(local_tid_17708, 32) * 32) &&
                        ltid_in_bounds_17733) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)] =
                                x_17719;
                            x_17721 = x_17719;
                            ((volatile __local
                              int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)] =
                                x_17720;
                            x_17722 = x_17720;
                        }
                    }
                    if (sle32(wave_sizze_17710, skip_threads_17734)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17734 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17708 - squot32(local_tid_17708, 32) * 32) ==
                    31 && ltid_in_bounds_17733) {
                    ((volatile __local
                      bool *) scan_arr_mem_17712)[sext_i32_i64(squot32(local_tid_17708,
                                                                       32))] =
                        x_17719;
                    ((volatile __local
                      int64_t *) scan_arr_mem_17714)[sext_i32_i64(squot32(local_tid_17708,
                                                                          32))] =
                        x_17720;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17735;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17708, 32) == 0 &&
                        ltid_in_bounds_17733) {
                        x_17728 = ((volatile __local
                                    bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)];
                        x_17729 = ((volatile __local
                                    int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)];
                        if ((local_tid_17708 - squot32(local_tid_17708, 32) *
                             32) == 0) {
                            x_17726 = x_17728;
                            x_17727 = x_17729;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17735 = 1;
                    while (slt32(skip_threads_17735, 32)) {
                        if (sle32(skip_threads_17735, local_tid_17708 -
                                  squot32(local_tid_17708, 32) * 32) &&
                            (squot32(local_tid_17708, 32) == 0 &&
                             ltid_in_bounds_17733)) {
                            // read operands
                            {
                                x_17726 = ((volatile __local
                                            bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708) -
                                                                        sext_i32_i64(skip_threads_17735)];
                                x_17727 = ((volatile __local
                                            int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708) -
                                                                           sext_i32_i64(skip_threads_17735)];
                            }
                            // perform operation
                            {
                                bool res_17730 = x_17726 || x_17728;
                                int64_t res_17731;
                                
                                if (x_17728) {
                                    res_17731 = x_17729;
                                } else {
                                    int64_t res_17732 = add64(x_17727, x_17729);
                                    
                                    res_17731 = res_17732;
                                }
                                x_17726 = res_17730;
                                x_17727 = res_17731;
                            }
                        }
                        if (sle32(wave_sizze_17710, skip_threads_17735)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17735, local_tid_17708 -
                                  squot32(local_tid_17708, 32) * 32) &&
                            (squot32(local_tid_17708, 32) == 0 &&
                             ltid_in_bounds_17733)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)] =
                                    x_17726;
                                x_17728 = x_17726;
                                ((volatile __local
                                  int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)] =
                                    x_17727;
                                x_17729 = x_17727;
                            }
                        }
                        if (sle32(wave_sizze_17710, skip_threads_17735)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17735 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17708, 32) == 0 ||
                      !ltid_in_bounds_17733)) {
                    // read operands
                    {
                        x_17721 = x_17719;
                        x_17722 = x_17720;
                        x_17719 = ((__local
                                    bool *) scan_arr_mem_17712)[sext_i32_i64(squot32(local_tid_17708,
                                                                                     32)) -
                                                                1];
                        x_17720 = ((__local
                                    int64_t *) scan_arr_mem_17714)[sext_i32_i64(squot32(local_tid_17708,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        bool res_17723 = x_17719 || x_17721;
                        int64_t res_17724;
                        
                        if (x_17721) {
                            res_17724 = x_17722;
                        } else {
                            int64_t res_17725 = add64(x_17720, x_17722);
                            
                            res_17724 = res_17725;
                        }
                        x_17719 = res_17723;
                        x_17720 = res_17724;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)] =
                            x_17719;
                        ((__local
                          int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)] =
                            x_17720;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17708, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)] =
                        x_17721;
                    ((__local
                      int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)] =
                        x_17722;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_17202, res_16524)) {
                    ((__global bool *) mem_17359)[gtid_17202] = ((__local
                                                                  bool *) scan_arr_mem_17712)[sext_i32_i64(local_tid_17708)];
                    ((__global int64_t *) mem_17361)[gtid_17202] = ((__local
                                                                     int64_t *) scan_arr_mem_17714)[sext_i32_i64(local_tid_17708)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17736 = 0;
                bool should_load_carry_17737 = local_tid_17708 == 0 &&
                     !crosses_segment_17736;
                
                if (should_load_carry_17737) {
                    x_16579 = ((__local
                                bool *) scan_arr_mem_17712)[segscan_group_sizze_17198 -
                                                            1];
                    x_16580 = ((__local
                                int64_t *) scan_arr_mem_17714)[segscan_group_sizze_17198 -
                                                               1];
                }
                if (!should_load_carry_17737) {
                    x_16579 = 0;
                    x_16580 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_17198
}
__kernel void mainziscan_stage1_17211(__global int *global_failure,
                                      int failure_is_an_option, __global
                                      int64_t *global_failure_args,
                                      __local volatile
                                      int64_t *scan_arr_mem_17781_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_17779_backing_aligned_1,
                                      int64_t paths_16408,
                                      float swap_term_16410,
                                      int64_t payments_16411,
                                      float notional_16412, float a_16413,
                                      float b_16414, float sigma_16415,
                                      float res_16511, int64_t res_16524,
                                      int64_t i_17163, __global
                                      unsigned char *mem_17336, __global
                                      unsigned char *mem_17356, __global
                                      unsigned char *mem_17361, __global
                                      unsigned char *mem_17363, __global
                                      unsigned char *mem_17366, __global
                                      unsigned char *mem_17368,
                                      int32_t num_threads_17773)
{
    #define segscan_group_sizze_17206 (mainzisegscan_group_sizze_17205)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17781_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17781_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17779_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17779_backing_aligned_1;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_17774;
    int32_t local_tid_17775;
    int64_t group_sizze_17778;
    int32_t wave_sizze_17777;
    int32_t group_tid_17776;
    
    global_tid_17774 = get_global_id(0);
    local_tid_17775 = get_local_id(0);
    group_sizze_17778 = get_local_size(0);
    wave_sizze_17777 = LOCKSTEP_WIDTH;
    group_tid_17776 = get_group_id(0);
    
    int32_t phys_tid_17211;
    
    phys_tid_17211 = global_tid_17774;
    
    __local char *scan_arr_mem_17779;
    __local char *scan_arr_mem_17781;
    
    scan_arr_mem_17779 = (__local char *) scan_arr_mem_17779_backing_0;
    scan_arr_mem_17781 = (__local char *) scan_arr_mem_17781_backing_1;
    
    bool x_16605;
    float x_16606;
    bool x_16607;
    float x_16608;
    
    x_16605 = 0;
    x_16606 = 0.0F;
    for (int64_t j_17783 = 0; j_17783 < sdiv_up64(res_16524,
                                                  sext_i32_i64(num_threads_17773));
         j_17783++) {
        int64_t chunk_offset_17784 = segscan_group_sizze_17206 * j_17783 +
                sext_i32_i64(group_tid_17776) * (segscan_group_sizze_17206 *
                                                 sdiv_up64(res_16524,
                                                           sext_i32_i64(num_threads_17773)));
        int64_t flat_idx_17785 = chunk_offset_17784 +
                sext_i32_i64(local_tid_17775);
        int64_t gtid_17210 = flat_idx_17785;
        
        // threads in bounds read input
        {
            if (slt64(gtid_17210, res_16524)) {
                int64_t x_16613 = ((__global int64_t *) mem_17361)[gtid_17210];
                int64_t x_16614 = ((__global int64_t *) mem_17356)[gtid_17210];
                bool x_16615 = ((__global bool *) mem_17363)[gtid_17210];
                int64_t res_16618 = sub64(x_16613, 1);
                bool x_16619 = sle64(0, x_16614);
                bool y_16620 = slt64(x_16614, paths_16408);
                bool bounds_check_16621 = x_16619 && y_16620;
                bool index_certs_16622;
                
                if (!bounds_check_16621) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 2) ==
                            -1) {
                            global_failure_args[0] = x_16614;
                            global_failure_args[1] = paths_16408;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float lifted_0_get_arg_16623 = ((__global
                                                 float *) mem_17336)[i_17163 *
                                                                     paths_16408 +
                                                                     x_16614];
                float ceil_arg_16624 = res_16511 / swap_term_16410;
                float res_16625;
                
                res_16625 = futrts_ceil32(ceil_arg_16624);
                
                float start_16626 = swap_term_16410 * res_16625;
                float ceil_arg_16627 = ceil_arg_16624 - 1.0F;
                float res_16628;
                
                res_16628 = futrts_ceil32(ceil_arg_16627);
                
                int64_t res_16629 = fptosi_f32_i64(res_16628);
                int64_t res_16630 = sub64(payments_16411, res_16629);
                int64_t sizze_16631 = sub64(res_16630, 1);
                bool cond_16632 = res_16618 == 0;
                float res_16633;
                
                if (cond_16632) {
                    res_16633 = 1.0F;
                } else {
                    res_16633 = 0.0F;
                }
                
                bool cond_16634 = slt64(0, res_16618);
                float res_16635;
                
                if (cond_16634) {
                    float y_16636 = 5.056644e-2F * swap_term_16410;
                    float res_16637 = res_16633 - y_16636;
                    
                    res_16635 = res_16637;
                } else {
                    res_16635 = res_16633;
                }
                
                bool cond_16638 = res_16618 == sizze_16631;
                float res_16639;
                
                if (cond_16638) {
                    float res_16640 = res_16635 - 1.0F;
                    
                    res_16639 = res_16640;
                } else {
                    res_16639 = res_16635;
                }
                
                float res_16641 = notional_16412 * res_16639;
                float res_16642 = sitofp_i64_f32(res_16618);
                float y_16643 = swap_term_16410 * res_16642;
                float bondprice_arg_16644 = start_16626 + y_16643;
                float y_16645 = bondprice_arg_16644 - res_16511;
                float negate_arg_16646 = a_16413 * y_16645;
                float exp_arg_16647 = 0.0F - negate_arg_16646;
                float res_16648 = fpow32(2.7182817F, exp_arg_16647);
                float x_16649 = 1.0F - res_16648;
                float B_16650 = x_16649 / a_16413;
                float x_16651 = B_16650 - bondprice_arg_16644;
                float x_16652 = res_16511 + x_16651;
                float x_16653 = fpow32(a_16413, 2.0F);
                float x_16654 = b_16414 * x_16653;
                float x_16655 = fpow32(sigma_16415, 2.0F);
                float y_16656 = x_16655 / 2.0F;
                float y_16657 = x_16654 - y_16656;
                float x_16658 = x_16652 * y_16657;
                float A1_16659 = x_16658 / x_16653;
                float y_16660 = fpow32(B_16650, 2.0F);
                float x_16661 = x_16655 * y_16660;
                float y_16662 = 4.0F * a_16413;
                float A2_16663 = x_16661 / y_16662;
                float exp_arg_16664 = A1_16659 - A2_16663;
                float res_16665 = fpow32(2.7182817F, exp_arg_16664);
                float negate_arg_16666 = lifted_0_get_arg_16623 * B_16650;
                float exp_arg_16667 = 0.0F - negate_arg_16666;
                float res_16668 = fpow32(2.7182817F, exp_arg_16667);
                float res_16669 = res_16665 * res_16668;
                float res_16670 = res_16641 * res_16669;
                
                // write to-scan values to parameters
                {
                    x_16607 = x_16615;
                    x_16608 = res_16670;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_17210, res_16524)) {
                    x_16607 = 0;
                    x_16608 = 0.0F;
                }
            }
            // combine with carry and write to local memory
            {
                bool res_16609 = x_16605 || x_16607;
                float res_16610;
                
                if (x_16607) {
                    res_16610 = x_16608;
                } else {
                    float res_16611 = x_16606 + x_16608;
                    
                    res_16610 = res_16611;
                }
                ((__local
                  bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)] =
                    res_16609;
                ((__local
                  float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)] =
                    res_16610;
            }
            
          error_0:
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_failure)
                return;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            bool x_17786;
            float x_17787;
            bool x_17788;
            float x_17789;
            bool x_17793;
            float x_17794;
            bool x_17795;
            float x_17796;
            bool ltid_in_bounds_17800;
            
            ltid_in_bounds_17800 = slt64(sext_i32_i64(local_tid_17775),
                                         segscan_group_sizze_17206);
            
            int32_t skip_threads_17801;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_17800) {
                    x_17788 = ((volatile __local
                                bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)];
                    x_17789 = ((volatile __local
                                float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)];
                    if ((local_tid_17775 - squot32(local_tid_17775, 32) * 32) ==
                        0) {
                        x_17786 = x_17788;
                        x_17787 = x_17789;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17801 = 1;
                while (slt32(skip_threads_17801, 32)) {
                    if (sle32(skip_threads_17801, local_tid_17775 -
                              squot32(local_tid_17775, 32) * 32) &&
                        ltid_in_bounds_17800) {
                        // read operands
                        {
                            x_17786 = ((volatile __local
                                        bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775) -
                                                                    sext_i32_i64(skip_threads_17801)];
                            x_17787 = ((volatile __local
                                        float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775) -
                                                                     sext_i32_i64(skip_threads_17801)];
                        }
                        // perform operation
                        {
                            bool res_17790 = x_17786 || x_17788;
                            float res_17791;
                            
                            if (x_17788) {
                                res_17791 = x_17789;
                            } else {
                                float res_17792 = x_17787 + x_17789;
                                
                                res_17791 = res_17792;
                            }
                            x_17786 = res_17790;
                            x_17787 = res_17791;
                        }
                    }
                    if (sle32(wave_sizze_17777, skip_threads_17801)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17801, local_tid_17775 -
                              squot32(local_tid_17775, 32) * 32) &&
                        ltid_in_bounds_17800) {
                        // write result
                        {
                            ((volatile __local
                              bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)] =
                                x_17786;
                            x_17788 = x_17786;
                            ((volatile __local
                              float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)] =
                                x_17787;
                            x_17789 = x_17787;
                        }
                    }
                    if (sle32(wave_sizze_17777, skip_threads_17801)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17801 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17775 - squot32(local_tid_17775, 32) * 32) ==
                    31 && ltid_in_bounds_17800) {
                    ((volatile __local
                      bool *) scan_arr_mem_17779)[sext_i32_i64(squot32(local_tid_17775,
                                                                       32))] =
                        x_17786;
                    ((volatile __local
                      float *) scan_arr_mem_17781)[sext_i32_i64(squot32(local_tid_17775,
                                                                        32))] =
                        x_17787;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17802;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17775, 32) == 0 &&
                        ltid_in_bounds_17800) {
                        x_17795 = ((volatile __local
                                    bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)];
                        x_17796 = ((volatile __local
                                    float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)];
                        if ((local_tid_17775 - squot32(local_tid_17775, 32) *
                             32) == 0) {
                            x_17793 = x_17795;
                            x_17794 = x_17796;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17802 = 1;
                    while (slt32(skip_threads_17802, 32)) {
                        if (sle32(skip_threads_17802, local_tid_17775 -
                                  squot32(local_tid_17775, 32) * 32) &&
                            (squot32(local_tid_17775, 32) == 0 &&
                             ltid_in_bounds_17800)) {
                            // read operands
                            {
                                x_17793 = ((volatile __local
                                            bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775) -
                                                                        sext_i32_i64(skip_threads_17802)];
                                x_17794 = ((volatile __local
                                            float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775) -
                                                                         sext_i32_i64(skip_threads_17802)];
                            }
                            // perform operation
                            {
                                bool res_17797 = x_17793 || x_17795;
                                float res_17798;
                                
                                if (x_17795) {
                                    res_17798 = x_17796;
                                } else {
                                    float res_17799 = x_17794 + x_17796;
                                    
                                    res_17798 = res_17799;
                                }
                                x_17793 = res_17797;
                                x_17794 = res_17798;
                            }
                        }
                        if (sle32(wave_sizze_17777, skip_threads_17802)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17802, local_tid_17775 -
                                  squot32(local_tid_17775, 32) * 32) &&
                            (squot32(local_tid_17775, 32) == 0 &&
                             ltid_in_bounds_17800)) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)] =
                                    x_17793;
                                x_17795 = x_17793;
                                ((volatile __local
                                  float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)] =
                                    x_17794;
                                x_17796 = x_17794;
                            }
                        }
                        if (sle32(wave_sizze_17777, skip_threads_17802)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17802 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17775, 32) == 0 ||
                      !ltid_in_bounds_17800)) {
                    // read operands
                    {
                        x_17788 = x_17786;
                        x_17789 = x_17787;
                        x_17786 = ((__local
                                    bool *) scan_arr_mem_17779)[sext_i32_i64(squot32(local_tid_17775,
                                                                                     32)) -
                                                                1];
                        x_17787 = ((__local
                                    float *) scan_arr_mem_17781)[sext_i32_i64(squot32(local_tid_17775,
                                                                                      32)) -
                                                                 1];
                    }
                    // perform operation
                    {
                        bool res_17790 = x_17786 || x_17788;
                        float res_17791;
                        
                        if (x_17788) {
                            res_17791 = x_17789;
                        } else {
                            float res_17792 = x_17787 + x_17789;
                            
                            res_17791 = res_17792;
                        }
                        x_17786 = res_17790;
                        x_17787 = res_17791;
                    }
                    // write final result
                    {
                        ((__local
                          bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)] =
                            x_17786;
                        ((__local
                          float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)] =
                            x_17787;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17775, 32) == 0) {
                    ((__local
                      bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)] =
                        x_17788;
                    ((__local
                      float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)] =
                        x_17789;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_17210, res_16524)) {
                    ((__global bool *) mem_17366)[gtid_17210] = ((__local
                                                                  bool *) scan_arr_mem_17779)[sext_i32_i64(local_tid_17775)];
                    ((__global float *) mem_17368)[gtid_17210] = ((__local
                                                                   float *) scan_arr_mem_17781)[sext_i32_i64(local_tid_17775)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17803 = 0;
                bool should_load_carry_17804 = local_tid_17775 == 0 &&
                     !crosses_segment_17803;
                
                if (should_load_carry_17804) {
                    x_16605 = ((__local
                                bool *) scan_arr_mem_17779)[segscan_group_sizze_17206 -
                                                            1];
                    x_16606 = ((__local
                                float *) scan_arr_mem_17781)[segscan_group_sizze_17206 -
                                                             1];
                }
                if (!should_load_carry_17804) {
                    x_16605 = 0;
                    x_16606 = 0.0F;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_17206
}
__kernel void mainziscan_stage1_17263(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17846_backing_aligned_0,
                                      int64_t res_16524, __global
                                      unsigned char *mem_17363, __global
                                      unsigned char *mem_17371,
                                      int32_t num_threads_17840)
{
    #define segscan_group_sizze_17258 (mainzisegscan_group_sizze_17257)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17846_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17846_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17841;
    int32_t local_tid_17842;
    int64_t group_sizze_17845;
    int32_t wave_sizze_17844;
    int32_t group_tid_17843;
    
    global_tid_17841 = get_global_id(0);
    local_tid_17842 = get_local_id(0);
    group_sizze_17845 = get_local_size(0);
    wave_sizze_17844 = LOCKSTEP_WIDTH;
    group_tid_17843 = get_group_id(0);
    
    int32_t phys_tid_17263;
    
    phys_tid_17263 = global_tid_17841;
    
    __local char *scan_arr_mem_17846;
    
    scan_arr_mem_17846 = (__local char *) scan_arr_mem_17846_backing_0;
    
    int64_t x_16694;
    int64_t x_16695;
    
    x_16694 = 0;
    for (int64_t j_17848 = 0; j_17848 < sdiv_up64(res_16524,
                                                  sext_i32_i64(num_threads_17840));
         j_17848++) {
        int64_t chunk_offset_17849 = segscan_group_sizze_17258 * j_17848 +
                sext_i32_i64(group_tid_17843) * (segscan_group_sizze_17258 *
                                                 sdiv_up64(res_16524,
                                                           sext_i32_i64(num_threads_17840)));
        int64_t flat_idx_17850 = chunk_offset_17849 +
                sext_i32_i64(local_tid_17842);
        int64_t gtid_17262 = flat_idx_17850;
        
        // threads in bounds read input
        {
            if (slt64(gtid_17262, res_16524)) {
                int64_t i_p_o_17291 = add64(1, gtid_17262);
                int64_t rot_i_17292 = smod64(i_p_o_17291, res_16524);
                bool x_16697 = ((__global bool *) mem_17363)[rot_i_17292];
                int64_t res_16698 = btoi_bool_i64(x_16697);
                
                // write to-scan values to parameters
                {
                    x_16695 = res_16698;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt64(gtid_17262, res_16524)) {
                    x_16695 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int64_t res_16696 = add64(x_16694, x_16695);
                
                ((__local
                  int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)] =
                    res_16696;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int64_t x_17851;
            int64_t x_17852;
            int64_t x_17854;
            int64_t x_17855;
            bool ltid_in_bounds_17857;
            
            ltid_in_bounds_17857 = slt64(sext_i32_i64(local_tid_17842),
                                         segscan_group_sizze_17258);
            
            int32_t skip_threads_17858;
            
            // read input for in-block scan
            {
                if (ltid_in_bounds_17857) {
                    x_17852 = ((volatile __local
                                int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)];
                    if ((local_tid_17842 - squot32(local_tid_17842, 32) * 32) ==
                        0) {
                        x_17851 = x_17852;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17858 = 1;
                while (slt32(skip_threads_17858, 32)) {
                    if (sle32(skip_threads_17858, local_tid_17842 -
                              squot32(local_tid_17842, 32) * 32) &&
                        ltid_in_bounds_17857) {
                        // read operands
                        {
                            x_17851 = ((volatile __local
                                        int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842) -
                                                                       sext_i32_i64(skip_threads_17858)];
                        }
                        // perform operation
                        {
                            int64_t res_17853 = add64(x_17851, x_17852);
                            
                            x_17851 = res_17853;
                        }
                    }
                    if (sle32(wave_sizze_17844, skip_threads_17858)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17858, local_tid_17842 -
                              squot32(local_tid_17842, 32) * 32) &&
                        ltid_in_bounds_17857) {
                        // write result
                        {
                            ((volatile __local
                              int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)] =
                                x_17851;
                            x_17852 = x_17851;
                        }
                    }
                    if (sle32(wave_sizze_17844, skip_threads_17858)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17858 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17842 - squot32(local_tid_17842, 32) * 32) ==
                    31 && ltid_in_bounds_17857) {
                    ((volatile __local
                      int64_t *) scan_arr_mem_17846)[sext_i32_i64(squot32(local_tid_17842,
                                                                          32))] =
                        x_17851;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17859;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17842, 32) == 0 &&
                        ltid_in_bounds_17857) {
                        x_17855 = ((volatile __local
                                    int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)];
                        if ((local_tid_17842 - squot32(local_tid_17842, 32) *
                             32) == 0) {
                            x_17854 = x_17855;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17859 = 1;
                    while (slt32(skip_threads_17859, 32)) {
                        if (sle32(skip_threads_17859, local_tid_17842 -
                                  squot32(local_tid_17842, 32) * 32) &&
                            (squot32(local_tid_17842, 32) == 0 &&
                             ltid_in_bounds_17857)) {
                            // read operands
                            {
                                x_17854 = ((volatile __local
                                            int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842) -
                                                                           sext_i32_i64(skip_threads_17859)];
                            }
                            // perform operation
                            {
                                int64_t res_17856 = add64(x_17854, x_17855);
                                
                                x_17854 = res_17856;
                            }
                        }
                        if (sle32(wave_sizze_17844, skip_threads_17859)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17859, local_tid_17842 -
                                  squot32(local_tid_17842, 32) * 32) &&
                            (squot32(local_tid_17842, 32) == 0 &&
                             ltid_in_bounds_17857)) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)] =
                                    x_17854;
                                x_17855 = x_17854;
                            }
                        }
                        if (sle32(wave_sizze_17844, skip_threads_17859)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17859 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17842, 32) == 0 ||
                      !ltid_in_bounds_17857)) {
                    // read operands
                    {
                        x_17852 = x_17851;
                        x_17851 = ((__local
                                    int64_t *) scan_arr_mem_17846)[sext_i32_i64(squot32(local_tid_17842,
                                                                                        32)) -
                                                                   1];
                    }
                    // perform operation
                    {
                        int64_t res_17853 = add64(x_17851, x_17852);
                        
                        x_17851 = res_17853;
                    }
                    // write final result
                    {
                        ((__local
                          int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)] =
                            x_17851;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17842, 32) == 0) {
                    ((__local
                      int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)] =
                        x_17852;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt64(gtid_17262, res_16524)) {
                    ((__global int64_t *) mem_17371)[gtid_17262] = ((__local
                                                                     int64_t *) scan_arr_mem_17846)[sext_i32_i64(local_tid_17842)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17860 = 0;
                bool should_load_carry_17861 = local_tid_17842 == 0 &&
                     !crosses_segment_17860;
                
                if (should_load_carry_17861) {
                    x_16694 = ((__local
                                int64_t *) scan_arr_mem_17846)[segscan_group_sizze_17258 -
                                                               1];
                }
                if (!should_load_carry_17861) {
                    x_16694 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_17258
}
__kernel void mainziscan_stage2_17172(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17459_backing_aligned_0,
                                      int64_t paths_16408, __global
                                      unsigned char *mem_17348,
                                      int64_t stage1_num_groups_17431,
                                      int32_t num_threads_17432)
{
    #define segscan_group_sizze_17167 (mainzisegscan_group_sizze_17166)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17459_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17459_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17454;
    int32_t local_tid_17455;
    int64_t group_sizze_17458;
    int32_t wave_sizze_17457;
    int32_t group_tid_17456;
    
    global_tid_17454 = get_global_id(0);
    local_tid_17455 = get_local_id(0);
    group_sizze_17458 = get_local_size(0);
    wave_sizze_17457 = LOCKSTEP_WIDTH;
    group_tid_17456 = get_group_id(0);
    
    int32_t phys_tid_17172;
    
    phys_tid_17172 = global_tid_17454;
    
    __local char *scan_arr_mem_17459;
    
    scan_arr_mem_17459 = (__local char *) scan_arr_mem_17459_backing_0;
    
    int64_t flat_idx_17461;
    
    flat_idx_17461 = (sext_i32_i64(local_tid_17455) + 1) *
        (segscan_group_sizze_17167 * sdiv_up64(paths_16408,
                                               sext_i32_i64(num_threads_17432))) -
        1;
    
    int64_t gtid_17171;
    
    gtid_17171 = flat_idx_17461;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_17171, paths_16408)) {
            ((__local
              int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)] =
                ((__global int64_t *) mem_17348)[gtid_17171];
        } else {
            ((__local
              int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int64_t x_16519;
    int64_t x_16520;
    int64_t x_17462;
    int64_t x_17463;
    bool ltid_in_bounds_17465;
    
    ltid_in_bounds_17465 = slt64(sext_i32_i64(local_tid_17455),
                                 stage1_num_groups_17431);
    
    int32_t skip_threads_17466;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_17465) {
            x_16520 = ((volatile __local
                        int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)];
            if ((local_tid_17455 - squot32(local_tid_17455, 32) * 32) == 0) {
                x_16519 = x_16520;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17466 = 1;
        while (slt32(skip_threads_17466, 32)) {
            if (sle32(skip_threads_17466, local_tid_17455 -
                      squot32(local_tid_17455, 32) * 32) &&
                ltid_in_bounds_17465) {
                // read operands
                {
                    x_16519 = ((volatile __local
                                int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455) -
                                                               sext_i32_i64(skip_threads_17466)];
                }
                // perform operation
                {
                    int64_t res_16521 = add64(x_16519, x_16520);
                    
                    x_16519 = res_16521;
                }
            }
            if (sle32(wave_sizze_17457, skip_threads_17466)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17466, local_tid_17455 -
                      squot32(local_tid_17455, 32) * 32) &&
                ltid_in_bounds_17465) {
                // write result
                {
                    ((volatile __local
                      int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)] =
                        x_16519;
                    x_16520 = x_16519;
                }
            }
            if (sle32(wave_sizze_17457, skip_threads_17466)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17466 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17455 - squot32(local_tid_17455, 32) * 32) == 31 &&
            ltid_in_bounds_17465) {
            ((volatile __local
              int64_t *) scan_arr_mem_17459)[sext_i32_i64(squot32(local_tid_17455,
                                                                  32))] =
                x_16519;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17467;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17455, 32) == 0 && ltid_in_bounds_17465) {
                x_17463 = ((volatile __local
                            int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)];
                if ((local_tid_17455 - squot32(local_tid_17455, 32) * 32) ==
                    0) {
                    x_17462 = x_17463;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17467 = 1;
            while (slt32(skip_threads_17467, 32)) {
                if (sle32(skip_threads_17467, local_tid_17455 -
                          squot32(local_tid_17455, 32) * 32) &&
                    (squot32(local_tid_17455, 32) == 0 &&
                     ltid_in_bounds_17465)) {
                    // read operands
                    {
                        x_17462 = ((volatile __local
                                    int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455) -
                                                                   sext_i32_i64(skip_threads_17467)];
                    }
                    // perform operation
                    {
                        int64_t res_17464 = add64(x_17462, x_17463);
                        
                        x_17462 = res_17464;
                    }
                }
                if (sle32(wave_sizze_17457, skip_threads_17467)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17467, local_tid_17455 -
                          squot32(local_tid_17455, 32) * 32) &&
                    (squot32(local_tid_17455, 32) == 0 &&
                     ltid_in_bounds_17465)) {
                    // write result
                    {
                        ((volatile __local
                          int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)] =
                            x_17462;
                        x_17463 = x_17462;
                    }
                }
                if (sle32(wave_sizze_17457, skip_threads_17467)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17467 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17455, 32) == 0 || !ltid_in_bounds_17465)) {
            // read operands
            {
                x_16520 = x_16519;
                x_16519 = ((__local
                            int64_t *) scan_arr_mem_17459)[sext_i32_i64(squot32(local_tid_17455,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                int64_t res_16521 = add64(x_16519, x_16520);
                
                x_16519 = res_16521;
            }
            // write final result
            {
                ((__local
                  int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)] =
                    x_16519;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17455, 32) == 0) {
            ((__local
              int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)] =
                x_16520;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_17171, paths_16408)) {
            ((__global int64_t *) mem_17348)[gtid_17171] = ((__local
                                                             int64_t *) scan_arr_mem_17459)[sext_i32_i64(local_tid_17455)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17167
}
__kernel void mainziscan_stage2_17195(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17678_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_17676_backing_aligned_1,
                                      int64_t res_16524, __global
                                      unsigned char *mem_17354, __global
                                      unsigned char *mem_17356,
                                      int64_t stage1_num_groups_17638,
                                      int32_t num_threads_17639)
{
    #define segscan_group_sizze_17190 (mainzisegscan_group_sizze_17189)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17678_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17678_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17676_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17676_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17671;
    int32_t local_tid_17672;
    int64_t group_sizze_17675;
    int32_t wave_sizze_17674;
    int32_t group_tid_17673;
    
    global_tid_17671 = get_global_id(0);
    local_tid_17672 = get_local_id(0);
    group_sizze_17675 = get_local_size(0);
    wave_sizze_17674 = LOCKSTEP_WIDTH;
    group_tid_17673 = get_group_id(0);
    
    int32_t phys_tid_17195;
    
    phys_tid_17195 = global_tid_17671;
    
    __local char *scan_arr_mem_17676;
    __local char *scan_arr_mem_17678;
    
    scan_arr_mem_17676 = (__local char *) scan_arr_mem_17676_backing_0;
    scan_arr_mem_17678 = (__local char *) scan_arr_mem_17678_backing_1;
    
    int64_t flat_idx_17680;
    
    flat_idx_17680 = (sext_i32_i64(local_tid_17672) + 1) *
        (segscan_group_sizze_17190 * sdiv_up64(res_16524,
                                               sext_i32_i64(num_threads_17639))) -
        1;
    
    int64_t gtid_17194;
    
    gtid_17194 = flat_idx_17680;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_17194, res_16524)) {
            ((__local
              bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)] =
                ((__global bool *) mem_17354)[gtid_17194];
            ((__local
              int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)] =
                ((__global int64_t *) mem_17356)[gtid_17194];
        } else {
            ((__local
              bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)] = 0;
            ((__local
              int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_16540;
    int64_t x_16541;
    bool x_16542;
    int64_t x_16543;
    bool x_17681;
    int64_t x_17682;
    bool x_17683;
    int64_t x_17684;
    bool ltid_in_bounds_17688;
    
    ltid_in_bounds_17688 = slt64(sext_i32_i64(local_tid_17672),
                                 stage1_num_groups_17638);
    
    int32_t skip_threads_17689;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_17688) {
            x_16542 = ((volatile __local
                        bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)];
            x_16543 = ((volatile __local
                        int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)];
            if ((local_tid_17672 - squot32(local_tid_17672, 32) * 32) == 0) {
                x_16540 = x_16542;
                x_16541 = x_16543;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17689 = 1;
        while (slt32(skip_threads_17689, 32)) {
            if (sle32(skip_threads_17689, local_tid_17672 -
                      squot32(local_tid_17672, 32) * 32) &&
                ltid_in_bounds_17688) {
                // read operands
                {
                    x_16540 = ((volatile __local
                                bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672) -
                                                            sext_i32_i64(skip_threads_17689)];
                    x_16541 = ((volatile __local
                                int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672) -
                                                               sext_i32_i64(skip_threads_17689)];
                }
                // perform operation
                {
                    bool res_16544 = x_16540 || x_16542;
                    int64_t res_16545;
                    
                    if (x_16542) {
                        res_16545 = x_16543;
                    } else {
                        int64_t res_16546 = add64(x_16541, x_16543);
                        
                        res_16545 = res_16546;
                    }
                    x_16540 = res_16544;
                    x_16541 = res_16545;
                }
            }
            if (sle32(wave_sizze_17674, skip_threads_17689)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17689, local_tid_17672 -
                      squot32(local_tid_17672, 32) * 32) &&
                ltid_in_bounds_17688) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)] =
                        x_16540;
                    x_16542 = x_16540;
                    ((volatile __local
                      int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)] =
                        x_16541;
                    x_16543 = x_16541;
                }
            }
            if (sle32(wave_sizze_17674, skip_threads_17689)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17689 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17672 - squot32(local_tid_17672, 32) * 32) == 31 &&
            ltid_in_bounds_17688) {
            ((volatile __local
              bool *) scan_arr_mem_17676)[sext_i32_i64(squot32(local_tid_17672,
                                                               32))] = x_16540;
            ((volatile __local
              int64_t *) scan_arr_mem_17678)[sext_i32_i64(squot32(local_tid_17672,
                                                                  32))] =
                x_16541;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17690;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17672, 32) == 0 && ltid_in_bounds_17688) {
                x_17683 = ((volatile __local
                            bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)];
                x_17684 = ((volatile __local
                            int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)];
                if ((local_tid_17672 - squot32(local_tid_17672, 32) * 32) ==
                    0) {
                    x_17681 = x_17683;
                    x_17682 = x_17684;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17690 = 1;
            while (slt32(skip_threads_17690, 32)) {
                if (sle32(skip_threads_17690, local_tid_17672 -
                          squot32(local_tid_17672, 32) * 32) &&
                    (squot32(local_tid_17672, 32) == 0 &&
                     ltid_in_bounds_17688)) {
                    // read operands
                    {
                        x_17681 = ((volatile __local
                                    bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672) -
                                                                sext_i32_i64(skip_threads_17690)];
                        x_17682 = ((volatile __local
                                    int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672) -
                                                                   sext_i32_i64(skip_threads_17690)];
                    }
                    // perform operation
                    {
                        bool res_17685 = x_17681 || x_17683;
                        int64_t res_17686;
                        
                        if (x_17683) {
                            res_17686 = x_17684;
                        } else {
                            int64_t res_17687 = add64(x_17682, x_17684);
                            
                            res_17686 = res_17687;
                        }
                        x_17681 = res_17685;
                        x_17682 = res_17686;
                    }
                }
                if (sle32(wave_sizze_17674, skip_threads_17690)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17690, local_tid_17672 -
                          squot32(local_tid_17672, 32) * 32) &&
                    (squot32(local_tid_17672, 32) == 0 &&
                     ltid_in_bounds_17688)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)] =
                            x_17681;
                        x_17683 = x_17681;
                        ((volatile __local
                          int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)] =
                            x_17682;
                        x_17684 = x_17682;
                    }
                }
                if (sle32(wave_sizze_17674, skip_threads_17690)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17690 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17672, 32) == 0 || !ltid_in_bounds_17688)) {
            // read operands
            {
                x_16542 = x_16540;
                x_16543 = x_16541;
                x_16540 = ((__local
                            bool *) scan_arr_mem_17676)[sext_i32_i64(squot32(local_tid_17672,
                                                                             32)) -
                                                        1];
                x_16541 = ((__local
                            int64_t *) scan_arr_mem_17678)[sext_i32_i64(squot32(local_tid_17672,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                bool res_16544 = x_16540 || x_16542;
                int64_t res_16545;
                
                if (x_16542) {
                    res_16545 = x_16543;
                } else {
                    int64_t res_16546 = add64(x_16541, x_16543);
                    
                    res_16545 = res_16546;
                }
                x_16540 = res_16544;
                x_16541 = res_16545;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)] =
                    x_16540;
                ((__local
                  int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)] =
                    x_16541;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17672, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)] =
                x_16542;
            ((__local
              int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)] =
                x_16543;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_17194, res_16524)) {
            ((__global bool *) mem_17354)[gtid_17194] = ((__local
                                                          bool *) scan_arr_mem_17676)[sext_i32_i64(local_tid_17672)];
            ((__global int64_t *) mem_17356)[gtid_17194] = ((__local
                                                             int64_t *) scan_arr_mem_17678)[sext_i32_i64(local_tid_17672)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17190
}
__kernel void mainziscan_stage2_17203(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17745_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_17743_backing_aligned_1,
                                      int64_t res_16524, __global
                                      unsigned char *mem_17359, __global
                                      unsigned char *mem_17361,
                                      int64_t stage1_num_groups_17705,
                                      int32_t num_threads_17706)
{
    #define segscan_group_sizze_17198 (mainzisegscan_group_sizze_17197)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17745_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17745_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17743_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17743_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17738;
    int32_t local_tid_17739;
    int64_t group_sizze_17742;
    int32_t wave_sizze_17741;
    int32_t group_tid_17740;
    
    global_tid_17738 = get_global_id(0);
    local_tid_17739 = get_local_id(0);
    group_sizze_17742 = get_local_size(0);
    wave_sizze_17741 = LOCKSTEP_WIDTH;
    group_tid_17740 = get_group_id(0);
    
    int32_t phys_tid_17203;
    
    phys_tid_17203 = global_tid_17738;
    
    __local char *scan_arr_mem_17743;
    __local char *scan_arr_mem_17745;
    
    scan_arr_mem_17743 = (__local char *) scan_arr_mem_17743_backing_0;
    scan_arr_mem_17745 = (__local char *) scan_arr_mem_17745_backing_1;
    
    int64_t flat_idx_17747;
    
    flat_idx_17747 = (sext_i32_i64(local_tid_17739) + 1) *
        (segscan_group_sizze_17198 * sdiv_up64(res_16524,
                                               sext_i32_i64(num_threads_17706))) -
        1;
    
    int64_t gtid_17202;
    
    gtid_17202 = flat_idx_17747;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_17202, res_16524)) {
            ((__local
              bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)] =
                ((__global bool *) mem_17359)[gtid_17202];
            ((__local
              int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)] =
                ((__global int64_t *) mem_17361)[gtid_17202];
        } else {
            ((__local
              bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)] = 0;
            ((__local
              int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_16579;
    int64_t x_16580;
    bool x_16581;
    int64_t x_16582;
    bool x_17748;
    int64_t x_17749;
    bool x_17750;
    int64_t x_17751;
    bool ltid_in_bounds_17755;
    
    ltid_in_bounds_17755 = slt64(sext_i32_i64(local_tid_17739),
                                 stage1_num_groups_17705);
    
    int32_t skip_threads_17756;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_17755) {
            x_16581 = ((volatile __local
                        bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)];
            x_16582 = ((volatile __local
                        int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)];
            if ((local_tid_17739 - squot32(local_tid_17739, 32) * 32) == 0) {
                x_16579 = x_16581;
                x_16580 = x_16582;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17756 = 1;
        while (slt32(skip_threads_17756, 32)) {
            if (sle32(skip_threads_17756, local_tid_17739 -
                      squot32(local_tid_17739, 32) * 32) &&
                ltid_in_bounds_17755) {
                // read operands
                {
                    x_16579 = ((volatile __local
                                bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739) -
                                                            sext_i32_i64(skip_threads_17756)];
                    x_16580 = ((volatile __local
                                int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739) -
                                                               sext_i32_i64(skip_threads_17756)];
                }
                // perform operation
                {
                    bool res_16583 = x_16579 || x_16581;
                    int64_t res_16584;
                    
                    if (x_16581) {
                        res_16584 = x_16582;
                    } else {
                        int64_t res_16585 = add64(x_16580, x_16582);
                        
                        res_16584 = res_16585;
                    }
                    x_16579 = res_16583;
                    x_16580 = res_16584;
                }
            }
            if (sle32(wave_sizze_17741, skip_threads_17756)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17756, local_tid_17739 -
                      squot32(local_tid_17739, 32) * 32) &&
                ltid_in_bounds_17755) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)] =
                        x_16579;
                    x_16581 = x_16579;
                    ((volatile __local
                      int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)] =
                        x_16580;
                    x_16582 = x_16580;
                }
            }
            if (sle32(wave_sizze_17741, skip_threads_17756)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17756 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17739 - squot32(local_tid_17739, 32) * 32) == 31 &&
            ltid_in_bounds_17755) {
            ((volatile __local
              bool *) scan_arr_mem_17743)[sext_i32_i64(squot32(local_tid_17739,
                                                               32))] = x_16579;
            ((volatile __local
              int64_t *) scan_arr_mem_17745)[sext_i32_i64(squot32(local_tid_17739,
                                                                  32))] =
                x_16580;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17757;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17739, 32) == 0 && ltid_in_bounds_17755) {
                x_17750 = ((volatile __local
                            bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)];
                x_17751 = ((volatile __local
                            int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)];
                if ((local_tid_17739 - squot32(local_tid_17739, 32) * 32) ==
                    0) {
                    x_17748 = x_17750;
                    x_17749 = x_17751;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17757 = 1;
            while (slt32(skip_threads_17757, 32)) {
                if (sle32(skip_threads_17757, local_tid_17739 -
                          squot32(local_tid_17739, 32) * 32) &&
                    (squot32(local_tid_17739, 32) == 0 &&
                     ltid_in_bounds_17755)) {
                    // read operands
                    {
                        x_17748 = ((volatile __local
                                    bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739) -
                                                                sext_i32_i64(skip_threads_17757)];
                        x_17749 = ((volatile __local
                                    int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739) -
                                                                   sext_i32_i64(skip_threads_17757)];
                    }
                    // perform operation
                    {
                        bool res_17752 = x_17748 || x_17750;
                        int64_t res_17753;
                        
                        if (x_17750) {
                            res_17753 = x_17751;
                        } else {
                            int64_t res_17754 = add64(x_17749, x_17751);
                            
                            res_17753 = res_17754;
                        }
                        x_17748 = res_17752;
                        x_17749 = res_17753;
                    }
                }
                if (sle32(wave_sizze_17741, skip_threads_17757)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17757, local_tid_17739 -
                          squot32(local_tid_17739, 32) * 32) &&
                    (squot32(local_tid_17739, 32) == 0 &&
                     ltid_in_bounds_17755)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)] =
                            x_17748;
                        x_17750 = x_17748;
                        ((volatile __local
                          int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)] =
                            x_17749;
                        x_17751 = x_17749;
                    }
                }
                if (sle32(wave_sizze_17741, skip_threads_17757)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17757 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17739, 32) == 0 || !ltid_in_bounds_17755)) {
            // read operands
            {
                x_16581 = x_16579;
                x_16582 = x_16580;
                x_16579 = ((__local
                            bool *) scan_arr_mem_17743)[sext_i32_i64(squot32(local_tid_17739,
                                                                             32)) -
                                                        1];
                x_16580 = ((__local
                            int64_t *) scan_arr_mem_17745)[sext_i32_i64(squot32(local_tid_17739,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                bool res_16583 = x_16579 || x_16581;
                int64_t res_16584;
                
                if (x_16581) {
                    res_16584 = x_16582;
                } else {
                    int64_t res_16585 = add64(x_16580, x_16582);
                    
                    res_16584 = res_16585;
                }
                x_16579 = res_16583;
                x_16580 = res_16584;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)] =
                    x_16579;
                ((__local
                  int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)] =
                    x_16580;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17739, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)] =
                x_16581;
            ((__local
              int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)] =
                x_16582;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_17202, res_16524)) {
            ((__global bool *) mem_17359)[gtid_17202] = ((__local
                                                          bool *) scan_arr_mem_17743)[sext_i32_i64(local_tid_17739)];
            ((__global int64_t *) mem_17361)[gtid_17202] = ((__local
                                                             int64_t *) scan_arr_mem_17745)[sext_i32_i64(local_tid_17739)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17198
}
__kernel void mainziscan_stage2_17211(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17812_backing_aligned_0,
                                      __local volatile
                                      int64_t *scan_arr_mem_17810_backing_aligned_1,
                                      int64_t res_16524, __global
                                      unsigned char *mem_17366, __global
                                      unsigned char *mem_17368,
                                      int64_t stage1_num_groups_17772,
                                      int32_t num_threads_17773)
{
    #define segscan_group_sizze_17206 (mainzisegscan_group_sizze_17205)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17812_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17812_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17810_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17810_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17805;
    int32_t local_tid_17806;
    int64_t group_sizze_17809;
    int32_t wave_sizze_17808;
    int32_t group_tid_17807;
    
    global_tid_17805 = get_global_id(0);
    local_tid_17806 = get_local_id(0);
    group_sizze_17809 = get_local_size(0);
    wave_sizze_17808 = LOCKSTEP_WIDTH;
    group_tid_17807 = get_group_id(0);
    
    int32_t phys_tid_17211;
    
    phys_tid_17211 = global_tid_17805;
    
    __local char *scan_arr_mem_17810;
    __local char *scan_arr_mem_17812;
    
    scan_arr_mem_17810 = (__local char *) scan_arr_mem_17810_backing_0;
    scan_arr_mem_17812 = (__local char *) scan_arr_mem_17812_backing_1;
    
    int64_t flat_idx_17814;
    
    flat_idx_17814 = (sext_i32_i64(local_tid_17806) + 1) *
        (segscan_group_sizze_17206 * sdiv_up64(res_16524,
                                               sext_i32_i64(num_threads_17773))) -
        1;
    
    int64_t gtid_17210;
    
    gtid_17210 = flat_idx_17814;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_17210, res_16524)) {
            ((__local
              bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)] =
                ((__global bool *) mem_17366)[gtid_17210];
            ((__local
              float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)] =
                ((__global float *) mem_17368)[gtid_17210];
        } else {
            ((__local
              bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)] = 0;
            ((__local
              float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)] =
                0.0F;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool x_16605;
    float x_16606;
    bool x_16607;
    float x_16608;
    bool x_17815;
    float x_17816;
    bool x_17817;
    float x_17818;
    bool ltid_in_bounds_17822;
    
    ltid_in_bounds_17822 = slt64(sext_i32_i64(local_tid_17806),
                                 stage1_num_groups_17772);
    
    int32_t skip_threads_17823;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_17822) {
            x_16607 = ((volatile __local
                        bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)];
            x_16608 = ((volatile __local
                        float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)];
            if ((local_tid_17806 - squot32(local_tid_17806, 32) * 32) == 0) {
                x_16605 = x_16607;
                x_16606 = x_16608;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17823 = 1;
        while (slt32(skip_threads_17823, 32)) {
            if (sle32(skip_threads_17823, local_tid_17806 -
                      squot32(local_tid_17806, 32) * 32) &&
                ltid_in_bounds_17822) {
                // read operands
                {
                    x_16605 = ((volatile __local
                                bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806) -
                                                            sext_i32_i64(skip_threads_17823)];
                    x_16606 = ((volatile __local
                                float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806) -
                                                             sext_i32_i64(skip_threads_17823)];
                }
                // perform operation
                {
                    bool res_16609 = x_16605 || x_16607;
                    float res_16610;
                    
                    if (x_16607) {
                        res_16610 = x_16608;
                    } else {
                        float res_16611 = x_16606 + x_16608;
                        
                        res_16610 = res_16611;
                    }
                    x_16605 = res_16609;
                    x_16606 = res_16610;
                }
            }
            if (sle32(wave_sizze_17808, skip_threads_17823)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17823, local_tid_17806 -
                      squot32(local_tid_17806, 32) * 32) &&
                ltid_in_bounds_17822) {
                // write result
                {
                    ((volatile __local
                      bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)] =
                        x_16605;
                    x_16607 = x_16605;
                    ((volatile __local
                      float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)] =
                        x_16606;
                    x_16608 = x_16606;
                }
            }
            if (sle32(wave_sizze_17808, skip_threads_17823)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17823 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17806 - squot32(local_tid_17806, 32) * 32) == 31 &&
            ltid_in_bounds_17822) {
            ((volatile __local
              bool *) scan_arr_mem_17810)[sext_i32_i64(squot32(local_tid_17806,
                                                               32))] = x_16605;
            ((volatile __local
              float *) scan_arr_mem_17812)[sext_i32_i64(squot32(local_tid_17806,
                                                                32))] = x_16606;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17824;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17806, 32) == 0 && ltid_in_bounds_17822) {
                x_17817 = ((volatile __local
                            bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)];
                x_17818 = ((volatile __local
                            float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)];
                if ((local_tid_17806 - squot32(local_tid_17806, 32) * 32) ==
                    0) {
                    x_17815 = x_17817;
                    x_17816 = x_17818;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17824 = 1;
            while (slt32(skip_threads_17824, 32)) {
                if (sle32(skip_threads_17824, local_tid_17806 -
                          squot32(local_tid_17806, 32) * 32) &&
                    (squot32(local_tid_17806, 32) == 0 &&
                     ltid_in_bounds_17822)) {
                    // read operands
                    {
                        x_17815 = ((volatile __local
                                    bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806) -
                                                                sext_i32_i64(skip_threads_17824)];
                        x_17816 = ((volatile __local
                                    float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806) -
                                                                 sext_i32_i64(skip_threads_17824)];
                    }
                    // perform operation
                    {
                        bool res_17819 = x_17815 || x_17817;
                        float res_17820;
                        
                        if (x_17817) {
                            res_17820 = x_17818;
                        } else {
                            float res_17821 = x_17816 + x_17818;
                            
                            res_17820 = res_17821;
                        }
                        x_17815 = res_17819;
                        x_17816 = res_17820;
                    }
                }
                if (sle32(wave_sizze_17808, skip_threads_17824)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17824, local_tid_17806 -
                          squot32(local_tid_17806, 32) * 32) &&
                    (squot32(local_tid_17806, 32) == 0 &&
                     ltid_in_bounds_17822)) {
                    // write result
                    {
                        ((volatile __local
                          bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)] =
                            x_17815;
                        x_17817 = x_17815;
                        ((volatile __local
                          float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)] =
                            x_17816;
                        x_17818 = x_17816;
                    }
                }
                if (sle32(wave_sizze_17808, skip_threads_17824)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17824 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17806, 32) == 0 || !ltid_in_bounds_17822)) {
            // read operands
            {
                x_16607 = x_16605;
                x_16608 = x_16606;
                x_16605 = ((__local
                            bool *) scan_arr_mem_17810)[sext_i32_i64(squot32(local_tid_17806,
                                                                             32)) -
                                                        1];
                x_16606 = ((__local
                            float *) scan_arr_mem_17812)[sext_i32_i64(squot32(local_tid_17806,
                                                                              32)) -
                                                         1];
            }
            // perform operation
            {
                bool res_16609 = x_16605 || x_16607;
                float res_16610;
                
                if (x_16607) {
                    res_16610 = x_16608;
                } else {
                    float res_16611 = x_16606 + x_16608;
                    
                    res_16610 = res_16611;
                }
                x_16605 = res_16609;
                x_16606 = res_16610;
            }
            // write final result
            {
                ((__local
                  bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)] =
                    x_16605;
                ((__local
                  float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)] =
                    x_16606;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17806, 32) == 0) {
            ((__local
              bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)] =
                x_16607;
            ((__local
              float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)] =
                x_16608;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_17210, res_16524)) {
            ((__global bool *) mem_17366)[gtid_17210] = ((__local
                                                          bool *) scan_arr_mem_17810)[sext_i32_i64(local_tid_17806)];
            ((__global float *) mem_17368)[gtid_17210] = ((__local
                                                           float *) scan_arr_mem_17812)[sext_i32_i64(local_tid_17806)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17206
}
__kernel void mainziscan_stage2_17263(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_17867_backing_aligned_0,
                                      int64_t res_16524, __global
                                      unsigned char *mem_17371,
                                      int64_t stage1_num_groups_17839,
                                      int32_t num_threads_17840)
{
    #define segscan_group_sizze_17258 (mainzisegscan_group_sizze_17257)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17867_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17867_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17862;
    int32_t local_tid_17863;
    int64_t group_sizze_17866;
    int32_t wave_sizze_17865;
    int32_t group_tid_17864;
    
    global_tid_17862 = get_global_id(0);
    local_tid_17863 = get_local_id(0);
    group_sizze_17866 = get_local_size(0);
    wave_sizze_17865 = LOCKSTEP_WIDTH;
    group_tid_17864 = get_group_id(0);
    
    int32_t phys_tid_17263;
    
    phys_tid_17263 = global_tid_17862;
    
    __local char *scan_arr_mem_17867;
    
    scan_arr_mem_17867 = (__local char *) scan_arr_mem_17867_backing_0;
    
    int64_t flat_idx_17869;
    
    flat_idx_17869 = (sext_i32_i64(local_tid_17863) + 1) *
        (segscan_group_sizze_17258 * sdiv_up64(res_16524,
                                               sext_i32_i64(num_threads_17840))) -
        1;
    
    int64_t gtid_17262;
    
    gtid_17262 = flat_idx_17869;
    // threads in bound read carries; others get neutral element
    {
        if (slt64(gtid_17262, res_16524)) {
            ((__local
              int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)] =
                ((__global int64_t *) mem_17371)[gtid_17262];
        } else {
            ((__local
              int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int64_t x_16694;
    int64_t x_16695;
    int64_t x_17870;
    int64_t x_17871;
    bool ltid_in_bounds_17873;
    
    ltid_in_bounds_17873 = slt64(sext_i32_i64(local_tid_17863),
                                 stage1_num_groups_17839);
    
    int32_t skip_threads_17874;
    
    // read input for in-block scan
    {
        if (ltid_in_bounds_17873) {
            x_16695 = ((volatile __local
                        int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)];
            if ((local_tid_17863 - squot32(local_tid_17863, 32) * 32) == 0) {
                x_16694 = x_16695;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17874 = 1;
        while (slt32(skip_threads_17874, 32)) {
            if (sle32(skip_threads_17874, local_tid_17863 -
                      squot32(local_tid_17863, 32) * 32) &&
                ltid_in_bounds_17873) {
                // read operands
                {
                    x_16694 = ((volatile __local
                                int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863) -
                                                               sext_i32_i64(skip_threads_17874)];
                }
                // perform operation
                {
                    int64_t res_16696 = add64(x_16694, x_16695);
                    
                    x_16694 = res_16696;
                }
            }
            if (sle32(wave_sizze_17865, skip_threads_17874)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17874, local_tid_17863 -
                      squot32(local_tid_17863, 32) * 32) &&
                ltid_in_bounds_17873) {
                // write result
                {
                    ((volatile __local
                      int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)] =
                        x_16694;
                    x_16695 = x_16694;
                }
            }
            if (sle32(wave_sizze_17865, skip_threads_17874)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17874 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17863 - squot32(local_tid_17863, 32) * 32) == 31 &&
            ltid_in_bounds_17873) {
            ((volatile __local
              int64_t *) scan_arr_mem_17867)[sext_i32_i64(squot32(local_tid_17863,
                                                                  32))] =
                x_16694;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17875;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17863, 32) == 0 && ltid_in_bounds_17873) {
                x_17871 = ((volatile __local
                            int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)];
                if ((local_tid_17863 - squot32(local_tid_17863, 32) * 32) ==
                    0) {
                    x_17870 = x_17871;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17875 = 1;
            while (slt32(skip_threads_17875, 32)) {
                if (sle32(skip_threads_17875, local_tid_17863 -
                          squot32(local_tid_17863, 32) * 32) &&
                    (squot32(local_tid_17863, 32) == 0 &&
                     ltid_in_bounds_17873)) {
                    // read operands
                    {
                        x_17870 = ((volatile __local
                                    int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863) -
                                                                   sext_i32_i64(skip_threads_17875)];
                    }
                    // perform operation
                    {
                        int64_t res_17872 = add64(x_17870, x_17871);
                        
                        x_17870 = res_17872;
                    }
                }
                if (sle32(wave_sizze_17865, skip_threads_17875)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17875, local_tid_17863 -
                          squot32(local_tid_17863, 32) * 32) &&
                    (squot32(local_tid_17863, 32) == 0 &&
                     ltid_in_bounds_17873)) {
                    // write result
                    {
                        ((volatile __local
                          int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)] =
                            x_17870;
                        x_17871 = x_17870;
                    }
                }
                if (sle32(wave_sizze_17865, skip_threads_17875)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17875 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17863, 32) == 0 || !ltid_in_bounds_17873)) {
            // read operands
            {
                x_16695 = x_16694;
                x_16694 = ((__local
                            int64_t *) scan_arr_mem_17867)[sext_i32_i64(squot32(local_tid_17863,
                                                                                32)) -
                                                           1];
            }
            // perform operation
            {
                int64_t res_16696 = add64(x_16694, x_16695);
                
                x_16694 = res_16696;
            }
            // write final result
            {
                ((__local
                  int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)] =
                    x_16694;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17863, 32) == 0) {
            ((__local
              int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)] =
                x_16695;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt64(gtid_17262, res_16524)) {
            ((__global int64_t *) mem_17371)[gtid_17262] = ((__local
                                                             int64_t *) scan_arr_mem_17867)[sext_i32_i64(local_tid_17863)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17258
}
__kernel void mainziscan_stage3_17172(__global int *global_failure,
                                      int64_t paths_16408,
                                      int64_t num_groups_17169, __global
                                      unsigned char *mem_17348,
                                      int32_t num_threads_17432,
                                      int32_t required_groups_17468)
{
    #define segscan_group_sizze_17167 (mainzisegscan_group_sizze_17166)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17469;
    int32_t local_tid_17470;
    int64_t group_sizze_17473;
    int32_t wave_sizze_17472;
    int32_t group_tid_17471;
    
    global_tid_17469 = get_global_id(0);
    local_tid_17470 = get_local_id(0);
    group_sizze_17473 = get_local_size(0);
    wave_sizze_17472 = LOCKSTEP_WIDTH;
    group_tid_17471 = get_group_id(0);
    
    int32_t phys_tid_17172;
    
    phys_tid_17172 = global_tid_17469;
    
    int32_t phys_group_id_17474;
    
    phys_group_id_17474 = get_group_id(0);
    for (int32_t i_17475 = 0; i_17475 < sdiv_up32(required_groups_17468 -
                                                  phys_group_id_17474,
                                                  sext_i64_i32(num_groups_17169));
         i_17475++) {
        int32_t virt_group_id_17476 = phys_group_id_17474 + i_17475 *
                sext_i64_i32(num_groups_17169);
        int64_t flat_idx_17477 = sext_i32_i64(virt_group_id_17476) *
                segscan_group_sizze_17167 + sext_i32_i64(local_tid_17470);
        int64_t gtid_17171 = flat_idx_17477;
        int64_t orig_group_17478 = squot64(flat_idx_17477,
                                           segscan_group_sizze_17167 *
                                           sdiv_up64(paths_16408,
                                                     sext_i32_i64(num_threads_17432)));
        int64_t carry_in_flat_idx_17479 = orig_group_17478 *
                (segscan_group_sizze_17167 * sdiv_up64(paths_16408,
                                                       sext_i32_i64(num_threads_17432))) -
                1;
        
        if (slt64(gtid_17171, paths_16408)) {
            if (!(orig_group_17478 == 0 || flat_idx_17477 == (orig_group_17478 +
                                                              1) *
                  (segscan_group_sizze_17167 * sdiv_up64(paths_16408,
                                                         sext_i32_i64(num_threads_17432))) -
                  1)) {
                int64_t x_16519;
                int64_t x_16520;
                
                x_16519 = ((__global
                            int64_t *) mem_17348)[carry_in_flat_idx_17479];
                x_16520 = ((__global int64_t *) mem_17348)[gtid_17171];
                
                int64_t res_16521;
                
                res_16521 = add64(x_16519, x_16520);
                x_16519 = res_16521;
                ((__global int64_t *) mem_17348)[gtid_17171] = x_16519;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17167
}
__kernel void mainziscan_stage3_17195(__global int *global_failure,
                                      int64_t res_16524,
                                      int64_t num_groups_17192, __global
                                      unsigned char *mem_17354, __global
                                      unsigned char *mem_17356,
                                      int32_t num_threads_17639,
                                      int32_t required_groups_17691)
{
    #define segscan_group_sizze_17190 (mainzisegscan_group_sizze_17189)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17692;
    int32_t local_tid_17693;
    int64_t group_sizze_17696;
    int32_t wave_sizze_17695;
    int32_t group_tid_17694;
    
    global_tid_17692 = get_global_id(0);
    local_tid_17693 = get_local_id(0);
    group_sizze_17696 = get_local_size(0);
    wave_sizze_17695 = LOCKSTEP_WIDTH;
    group_tid_17694 = get_group_id(0);
    
    int32_t phys_tid_17195;
    
    phys_tid_17195 = global_tid_17692;
    
    int32_t phys_group_id_17697;
    
    phys_group_id_17697 = get_group_id(0);
    for (int32_t i_17698 = 0; i_17698 < sdiv_up32(required_groups_17691 -
                                                  phys_group_id_17697,
                                                  sext_i64_i32(num_groups_17192));
         i_17698++) {
        int32_t virt_group_id_17699 = phys_group_id_17697 + i_17698 *
                sext_i64_i32(num_groups_17192);
        int64_t flat_idx_17700 = sext_i32_i64(virt_group_id_17699) *
                segscan_group_sizze_17190 + sext_i32_i64(local_tid_17693);
        int64_t gtid_17194 = flat_idx_17700;
        int64_t orig_group_17701 = squot64(flat_idx_17700,
                                           segscan_group_sizze_17190 *
                                           sdiv_up64(res_16524,
                                                     sext_i32_i64(num_threads_17639)));
        int64_t carry_in_flat_idx_17702 = orig_group_17701 *
                (segscan_group_sizze_17190 * sdiv_up64(res_16524,
                                                       sext_i32_i64(num_threads_17639))) -
                1;
        
        if (slt64(gtid_17194, res_16524)) {
            if (!(orig_group_17701 == 0 || flat_idx_17700 == (orig_group_17701 +
                                                              1) *
                  (segscan_group_sizze_17190 * sdiv_up64(res_16524,
                                                         sext_i32_i64(num_threads_17639))) -
                  1)) {
                bool x_16540;
                int64_t x_16541;
                bool x_16542;
                int64_t x_16543;
                
                x_16540 = ((__global
                            bool *) mem_17354)[carry_in_flat_idx_17702];
                x_16541 = ((__global
                            int64_t *) mem_17356)[carry_in_flat_idx_17702];
                x_16542 = ((__global bool *) mem_17354)[gtid_17194];
                x_16543 = ((__global int64_t *) mem_17356)[gtid_17194];
                
                bool res_16544;
                
                res_16544 = x_16540 || x_16542;
                
                int64_t res_16545;
                
                if (x_16542) {
                    res_16545 = x_16543;
                } else {
                    int64_t res_16546 = add64(x_16541, x_16543);
                    
                    res_16545 = res_16546;
                }
                x_16540 = res_16544;
                x_16541 = res_16545;
                ((__global bool *) mem_17354)[gtid_17194] = x_16540;
                ((__global int64_t *) mem_17356)[gtid_17194] = x_16541;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17190
}
__kernel void mainziscan_stage3_17203(__global int *global_failure,
                                      int64_t res_16524,
                                      int64_t num_groups_17200, __global
                                      unsigned char *mem_17359, __global
                                      unsigned char *mem_17361,
                                      int32_t num_threads_17706,
                                      int32_t required_groups_17758)
{
    #define segscan_group_sizze_17198 (mainzisegscan_group_sizze_17197)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17759;
    int32_t local_tid_17760;
    int64_t group_sizze_17763;
    int32_t wave_sizze_17762;
    int32_t group_tid_17761;
    
    global_tid_17759 = get_global_id(0);
    local_tid_17760 = get_local_id(0);
    group_sizze_17763 = get_local_size(0);
    wave_sizze_17762 = LOCKSTEP_WIDTH;
    group_tid_17761 = get_group_id(0);
    
    int32_t phys_tid_17203;
    
    phys_tid_17203 = global_tid_17759;
    
    int32_t phys_group_id_17764;
    
    phys_group_id_17764 = get_group_id(0);
    for (int32_t i_17765 = 0; i_17765 < sdiv_up32(required_groups_17758 -
                                                  phys_group_id_17764,
                                                  sext_i64_i32(num_groups_17200));
         i_17765++) {
        int32_t virt_group_id_17766 = phys_group_id_17764 + i_17765 *
                sext_i64_i32(num_groups_17200);
        int64_t flat_idx_17767 = sext_i32_i64(virt_group_id_17766) *
                segscan_group_sizze_17198 + sext_i32_i64(local_tid_17760);
        int64_t gtid_17202 = flat_idx_17767;
        int64_t orig_group_17768 = squot64(flat_idx_17767,
                                           segscan_group_sizze_17198 *
                                           sdiv_up64(res_16524,
                                                     sext_i32_i64(num_threads_17706)));
        int64_t carry_in_flat_idx_17769 = orig_group_17768 *
                (segscan_group_sizze_17198 * sdiv_up64(res_16524,
                                                       sext_i32_i64(num_threads_17706))) -
                1;
        
        if (slt64(gtid_17202, res_16524)) {
            if (!(orig_group_17768 == 0 || flat_idx_17767 == (orig_group_17768 +
                                                              1) *
                  (segscan_group_sizze_17198 * sdiv_up64(res_16524,
                                                         sext_i32_i64(num_threads_17706))) -
                  1)) {
                bool x_16579;
                int64_t x_16580;
                bool x_16581;
                int64_t x_16582;
                
                x_16579 = ((__global
                            bool *) mem_17359)[carry_in_flat_idx_17769];
                x_16580 = ((__global
                            int64_t *) mem_17361)[carry_in_flat_idx_17769];
                x_16581 = ((__global bool *) mem_17359)[gtid_17202];
                x_16582 = ((__global int64_t *) mem_17361)[gtid_17202];
                
                bool res_16583;
                
                res_16583 = x_16579 || x_16581;
                
                int64_t res_16584;
                
                if (x_16581) {
                    res_16584 = x_16582;
                } else {
                    int64_t res_16585 = add64(x_16580, x_16582);
                    
                    res_16584 = res_16585;
                }
                x_16579 = res_16583;
                x_16580 = res_16584;
                ((__global bool *) mem_17359)[gtid_17202] = x_16579;
                ((__global int64_t *) mem_17361)[gtid_17202] = x_16580;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17198
}
__kernel void mainziscan_stage3_17211(__global int *global_failure,
                                      int64_t res_16524,
                                      int64_t num_groups_17208, __global
                                      unsigned char *mem_17366, __global
                                      unsigned char *mem_17368,
                                      int32_t num_threads_17773,
                                      int32_t required_groups_17825)
{
    #define segscan_group_sizze_17206 (mainzisegscan_group_sizze_17205)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17826;
    int32_t local_tid_17827;
    int64_t group_sizze_17830;
    int32_t wave_sizze_17829;
    int32_t group_tid_17828;
    
    global_tid_17826 = get_global_id(0);
    local_tid_17827 = get_local_id(0);
    group_sizze_17830 = get_local_size(0);
    wave_sizze_17829 = LOCKSTEP_WIDTH;
    group_tid_17828 = get_group_id(0);
    
    int32_t phys_tid_17211;
    
    phys_tid_17211 = global_tid_17826;
    
    int32_t phys_group_id_17831;
    
    phys_group_id_17831 = get_group_id(0);
    for (int32_t i_17832 = 0; i_17832 < sdiv_up32(required_groups_17825 -
                                                  phys_group_id_17831,
                                                  sext_i64_i32(num_groups_17208));
         i_17832++) {
        int32_t virt_group_id_17833 = phys_group_id_17831 + i_17832 *
                sext_i64_i32(num_groups_17208);
        int64_t flat_idx_17834 = sext_i32_i64(virt_group_id_17833) *
                segscan_group_sizze_17206 + sext_i32_i64(local_tid_17827);
        int64_t gtid_17210 = flat_idx_17834;
        int64_t orig_group_17835 = squot64(flat_idx_17834,
                                           segscan_group_sizze_17206 *
                                           sdiv_up64(res_16524,
                                                     sext_i32_i64(num_threads_17773)));
        int64_t carry_in_flat_idx_17836 = orig_group_17835 *
                (segscan_group_sizze_17206 * sdiv_up64(res_16524,
                                                       sext_i32_i64(num_threads_17773))) -
                1;
        
        if (slt64(gtid_17210, res_16524)) {
            if (!(orig_group_17835 == 0 || flat_idx_17834 == (orig_group_17835 +
                                                              1) *
                  (segscan_group_sizze_17206 * sdiv_up64(res_16524,
                                                         sext_i32_i64(num_threads_17773))) -
                  1)) {
                bool x_16605;
                float x_16606;
                bool x_16607;
                float x_16608;
                
                x_16605 = ((__global
                            bool *) mem_17366)[carry_in_flat_idx_17836];
                x_16606 = ((__global
                            float *) mem_17368)[carry_in_flat_idx_17836];
                x_16607 = ((__global bool *) mem_17366)[gtid_17210];
                x_16608 = ((__global float *) mem_17368)[gtid_17210];
                
                bool res_16609;
                
                res_16609 = x_16605 || x_16607;
                
                float res_16610;
                
                if (x_16607) {
                    res_16610 = x_16608;
                } else {
                    float res_16611 = x_16606 + x_16608;
                    
                    res_16610 = res_16611;
                }
                x_16605 = res_16609;
                x_16606 = res_16610;
                ((__global bool *) mem_17366)[gtid_17210] = x_16605;
                ((__global float *) mem_17368)[gtid_17210] = x_16606;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17206
}
__kernel void mainziscan_stage3_17263(__global int *global_failure,
                                      int64_t res_16524,
                                      int64_t num_groups_17260, __global
                                      unsigned char *mem_17371,
                                      int32_t num_threads_17840,
                                      int32_t required_groups_17876)
{
    #define segscan_group_sizze_17258 (mainzisegscan_group_sizze_17257)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17877;
    int32_t local_tid_17878;
    int64_t group_sizze_17881;
    int32_t wave_sizze_17880;
    int32_t group_tid_17879;
    
    global_tid_17877 = get_global_id(0);
    local_tid_17878 = get_local_id(0);
    group_sizze_17881 = get_local_size(0);
    wave_sizze_17880 = LOCKSTEP_WIDTH;
    group_tid_17879 = get_group_id(0);
    
    int32_t phys_tid_17263;
    
    phys_tid_17263 = global_tid_17877;
    
    int32_t phys_group_id_17882;
    
    phys_group_id_17882 = get_group_id(0);
    for (int32_t i_17883 = 0; i_17883 < sdiv_up32(required_groups_17876 -
                                                  phys_group_id_17882,
                                                  sext_i64_i32(num_groups_17260));
         i_17883++) {
        int32_t virt_group_id_17884 = phys_group_id_17882 + i_17883 *
                sext_i64_i32(num_groups_17260);
        int64_t flat_idx_17885 = sext_i32_i64(virt_group_id_17884) *
                segscan_group_sizze_17258 + sext_i32_i64(local_tid_17878);
        int64_t gtid_17262 = flat_idx_17885;
        int64_t orig_group_17886 = squot64(flat_idx_17885,
                                           segscan_group_sizze_17258 *
                                           sdiv_up64(res_16524,
                                                     sext_i32_i64(num_threads_17840)));
        int64_t carry_in_flat_idx_17887 = orig_group_17886 *
                (segscan_group_sizze_17258 * sdiv_up64(res_16524,
                                                       sext_i32_i64(num_threads_17840))) -
                1;
        
        if (slt64(gtid_17262, res_16524)) {
            if (!(orig_group_17886 == 0 || flat_idx_17885 == (orig_group_17886 +
                                                              1) *
                  (segscan_group_sizze_17258 * sdiv_up64(res_16524,
                                                         sext_i32_i64(num_threads_17840))) -
                  1)) {
                int64_t x_16694;
                int64_t x_16695;
                
                x_16694 = ((__global
                            int64_t *) mem_17371)[carry_in_flat_idx_17887];
                x_16695 = ((__global int64_t *) mem_17371)[gtid_17262];
                
                int64_t res_16696;
                
                res_16696 = add64(x_16694, x_16695);
                x_16694 = res_16696;
                ((__global int64_t *) mem_17371)[gtid_17262] = x_16694;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_17258
}
__kernel void mainziseghist_global_17180(__global int *global_failure,
                                         int64_t paths_16408, int64_t res_16524,
                                         int64_t num_groups_17177, __global
                                         unsigned char *mem_17348,
                                         int32_t num_subhistos_17489, __global
                                         unsigned char *res_subhistos_mem_17490,
                                         __global
                                         unsigned char *mainzihist_locks_mem_17560,
                                         int32_t chk_i_17562,
                                         int64_t hist_H_chk_17563)
{
    #define seghist_group_sizze_17175 (mainziseghist_group_sizze_17174)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17564;
    int32_t local_tid_17565;
    int64_t group_sizze_17568;
    int32_t wave_sizze_17567;
    int32_t group_tid_17566;
    
    global_tid_17564 = get_global_id(0);
    local_tid_17565 = get_local_id(0);
    group_sizze_17568 = get_local_size(0);
    wave_sizze_17567 = LOCKSTEP_WIDTH;
    group_tid_17566 = get_group_id(0);
    
    int32_t phys_tid_17180;
    
    phys_tid_17180 = global_tid_17564;
    
    int32_t subhisto_ind_17569;
    
    subhisto_ind_17569 = squot32(global_tid_17564,
                                 sdiv_up32(sext_i64_i32(seghist_group_sizze_17175 *
                                           num_groups_17177),
                                           num_subhistos_17489));
    for (int64_t i_17570 = 0; i_17570 < sdiv_up64(paths_16408 -
                                                  sext_i32_i64(global_tid_17564),
                                                  sext_i32_i64(sext_i64_i32(seghist_group_sizze_17175 *
                                                  num_groups_17177)));
         i_17570++) {
        int32_t gtid_17179 = sext_i64_i32(i_17570 *
                sext_i32_i64(sext_i64_i32(seghist_group_sizze_17175 *
                num_groups_17177)) + sext_i32_i64(global_tid_17564));
        
        if (slt64(i_17570 *
                  sext_i32_i64(sext_i64_i32(seghist_group_sizze_17175 *
                  num_groups_17177)) + sext_i32_i64(global_tid_17564),
                  paths_16408)) {
            int64_t i_p_o_17285 = add64(-1, gtid_17179);
            int64_t rot_i_17286 = smod64(i_p_o_17285, paths_16408);
            bool cond_17186 = gtid_17179 == 0;
            int64_t res_17187;
            
            if (cond_17186) {
                res_17187 = 0;
            } else {
                int64_t x_17185 = ((__global int64_t *) mem_17348)[rot_i_17286];
                
                res_17187 = x_17185;
            }
            // save map-out results
            { }
            // perform atomic updates
            {
                if (sle64(sext_i32_i64(chk_i_17562) * hist_H_chk_17563,
                          res_17187) && (slt64(res_17187,
                                               sext_i32_i64(chk_i_17562) *
                                               hist_H_chk_17563 +
                                               hist_H_chk_17563) &&
                                         slt64(res_17187, res_16524))) {
                    int64_t x_17181;
                    int64_t x_17182;
                    
                    x_17182 = gtid_17179;
                    
                    int32_t old_17571;
                    volatile bool continue_17572;
                    
                    continue_17572 = 1;
                    while (continue_17572) {
                        old_17571 =
                            atomic_cmpxchg_i32_global(&((volatile __global
                                                         int *) mainzihist_locks_mem_17560)[srem64(sext_i32_i64(subhisto_ind_17569) *
                                                                                                   res_16524 +
                                                                                                   res_17187,
                                                                                                   100151)],
                                                      0, 1);
                        if (old_17571 == 0) {
                            int64_t x_17181;
                            
                            // bind lhs
                            {
                                x_17181 = ((volatile __global
                                            int64_t *) res_subhistos_mem_17490)[sext_i32_i64(subhisto_ind_17569) *
                                                                                res_16524 +
                                                                                res_17187];
                            }
                            // execute operation
                            {
                                int64_t res_17183 = smax64(x_17181, x_17182);
                                
                                x_17181 = res_17183;
                            }
                            // update global result
                            {
                                ((volatile __global
                                  int64_t *) res_subhistos_mem_17490)[sext_i32_i64(subhisto_ind_17569) *
                                                                      res_16524 +
                                                                      res_17187] =
                                    x_17181;
                            }
                            mem_fence_global();
                            old_17571 =
                                atomic_cmpxchg_i32_global(&((volatile __global
                                                             int *) mainzihist_locks_mem_17560)[srem64(sext_i32_i64(subhisto_ind_17569) *
                                                                                                       res_16524 +
                                                                                                       res_17187,
                                                                                                       100151)],
                                                          1, 0);
                            continue_17572 = 0;
                        }
                        mem_fence_global();
                    }
                }
            }
        }
    }
    
  error_0:
    return;
    #undef seghist_group_sizze_17175
}
__kernel void mainziseghist_local_17180(__global int *global_failure,
                                        __local volatile
                                        int64_t *locks_mem_17530_backing_aligned_0,
                                        __local volatile
                                        int64_t *subhistogram_local_mem_17528_backing_aligned_1,
                                        int64_t paths_16408, int64_t res_16524,
                                        __global unsigned char *mem_17348,
                                        __global
                                        unsigned char *res_subhistos_mem_17490,
                                        int32_t max_group_sizze_17499,
                                        int64_t num_groups_17500,
                                        int32_t hist_M_17506,
                                        int32_t chk_i_17511,
                                        int64_t num_segments_17512,
                                        int64_t hist_H_chk_17513,
                                        int64_t histo_sizze_17514,
                                        int32_t init_per_thread_17515)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict locks_mem_17530_backing_1 =
                          (__local volatile
                           char *) locks_mem_17530_backing_aligned_0;
    __local volatile char *restrict subhistogram_local_mem_17528_backing_0 =
                          (__local volatile
                           char *) subhistogram_local_mem_17528_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17516;
    int32_t local_tid_17517;
    int64_t group_sizze_17520;
    int32_t wave_sizze_17519;
    int32_t group_tid_17518;
    
    global_tid_17516 = get_global_id(0);
    local_tid_17517 = get_local_id(0);
    group_sizze_17520 = get_local_size(0);
    wave_sizze_17519 = LOCKSTEP_WIDTH;
    group_tid_17518 = get_group_id(0);
    
    int32_t phys_tid_17180;
    
    phys_tid_17180 = global_tid_17516;
    
    int32_t phys_group_id_17521;
    
    phys_group_id_17521 = get_group_id(0);
    for (int32_t i_17522 = 0; i_17522 <
         sdiv_up32(sext_i64_i32(num_groups_17500 * num_segments_17512) -
                   phys_group_id_17521, sext_i64_i32(num_groups_17500));
         i_17522++) {
        int32_t virt_group_id_17523 = phys_group_id_17521 + i_17522 *
                sext_i64_i32(num_groups_17500);
        int32_t flat_segment_id_17524 = squot32(virt_group_id_17523,
                                                sext_i64_i32(num_groups_17500));
        int32_t gid_in_segment_17525 = srem32(virt_group_id_17523,
                                              sext_i64_i32(num_groups_17500));
        int32_t pgtid_in_segment_17526 = gid_in_segment_17525 *
                sext_i64_i32(max_group_sizze_17499) + local_tid_17517;
        int32_t threads_per_segment_17527 = sext_i64_i32(num_groups_17500 *
                max_group_sizze_17499);
        __local char *subhistogram_local_mem_17528;
        
        subhistogram_local_mem_17528 = (__local
                                        char *) subhistogram_local_mem_17528_backing_0;
        
        __local char *locks_mem_17530;
        
        locks_mem_17530 = (__local char *) locks_mem_17530_backing_1;
        // All locks start out unlocked
        {
            for (int64_t i_17532 = 0; i_17532 < sdiv_up64(hist_M_17506 *
                                                          hist_H_chk_17513 -
                                                          sext_i32_i64(local_tid_17517),
                                                          max_group_sizze_17499);
                 i_17532++) {
                ((__local int32_t *) locks_mem_17530)[squot64(i_17532 *
                                                              max_group_sizze_17499 +
                                                              sext_i32_i64(local_tid_17517),
                                                              hist_H_chk_17513) *
                                                      hist_H_chk_17513 +
                                                      (i_17532 *
                                                       max_group_sizze_17499 +
                                                       sext_i32_i64(local_tid_17517) -
                                                       squot64(i_17532 *
                                                               max_group_sizze_17499 +
                                                               sext_i32_i64(local_tid_17517),
                                                               hist_H_chk_17513) *
                                                       hist_H_chk_17513)] = 0;
            }
        }
        
        int32_t thread_local_subhisto_i_17533;
        
        thread_local_subhisto_i_17533 = srem32(local_tid_17517, hist_M_17506);
        // initialize histograms in local memory
        {
            for (int32_t local_i_17534 = 0; local_i_17534 <
                 init_per_thread_17515; local_i_17534++) {
                int32_t j_17535 = local_i_17534 *
                        sext_i64_i32(max_group_sizze_17499) + local_tid_17517;
                int32_t j_offset_17536 = hist_M_17506 *
                        sext_i64_i32(histo_sizze_17514) * gid_in_segment_17525 +
                        j_17535;
                int32_t local_subhisto_i_17537 = squot32(j_17535,
                                                         sext_i64_i32(histo_sizze_17514));
                int32_t global_subhisto_i_17538 = squot32(j_offset_17536,
                                                          sext_i64_i32(histo_sizze_17514));
                
                if (slt32(j_17535, hist_M_17506 *
                          sext_i64_i32(histo_sizze_17514))) {
                    // First subhistogram is initialised from global memory; others with neutral element.
                    {
                        if (global_subhisto_i_17538 == 0) {
                            ((__local
                              int64_t *) subhistogram_local_mem_17528)[sext_i32_i64(local_subhisto_i_17537) *
                                                                       hist_H_chk_17513 +
                                                                       sext_i32_i64(srem32(j_17535,
                                                                                           sext_i64_i32(histo_sizze_17514)))] =
                                ((__global
                                  int64_t *) res_subhistos_mem_17490)[sext_i32_i64(srem32(j_17535,
                                                                                          sext_i64_i32(histo_sizze_17514))) +
                                                                      sext_i32_i64(chk_i_17511) *
                                                                      hist_H_chk_17513];
                        } else {
                            ((__local
                              int64_t *) subhistogram_local_mem_17528)[sext_i32_i64(local_subhisto_i_17537) *
                                                                       hist_H_chk_17513 +
                                                                       sext_i32_i64(srem32(j_17535,
                                                                                           sext_i64_i32(histo_sizze_17514)))] =
                                0;
                        }
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_17539 = 0; i_17539 <
             sdiv_up32(sext_i64_i32(paths_16408) - pgtid_in_segment_17526,
                       threads_per_segment_17527); i_17539++) {
            int32_t gtid_17179 = i_17539 * threads_per_segment_17527 +
                    pgtid_in_segment_17526;
            int64_t i_p_o_17285 = add64(-1, gtid_17179);
            int64_t rot_i_17286 = smod64(i_p_o_17285, paths_16408);
            bool cond_17186 = gtid_17179 == 0;
            int64_t res_17187;
            
            if (cond_17186) {
                res_17187 = 0;
            } else {
                int64_t x_17185 = ((__global int64_t *) mem_17348)[rot_i_17286];
                
                res_17187 = x_17185;
            }
            if (chk_i_17511 == 0) {
                // save map-out results
                { }
            }
            // perform atomic updates
            {
                if (slt64(res_17187, res_16524) &&
                    (sle64(sext_i32_i64(chk_i_17511) * hist_H_chk_17513,
                           res_17187) && slt64(res_17187,
                                               sext_i32_i64(chk_i_17511) *
                                               hist_H_chk_17513 +
                                               hist_H_chk_17513))) {
                    int64_t x_17181;
                    int64_t x_17182;
                    
                    x_17182 = gtid_17179;
                    
                    int32_t old_17540;
                    volatile bool continue_17541;
                    
                    continue_17541 = 1;
                    while (continue_17541) {
                        old_17540 = atomic_cmpxchg_i32_local(&((volatile __local
                                                                int *) locks_mem_17530)[sext_i32_i64(thread_local_subhisto_i_17533) *
                                                                                        hist_H_chk_17513 +
                                                                                        (res_17187 -
                                                                                         sext_i32_i64(chk_i_17511) *
                                                                                         hist_H_chk_17513)],
                                                             0, 1);
                        if (old_17540 == 0) {
                            int64_t x_17181;
                            
                            // bind lhs
                            {
                                x_17181 = ((volatile __local
                                            int64_t *) subhistogram_local_mem_17528)[sext_i32_i64(thread_local_subhisto_i_17533) *
                                                                                     hist_H_chk_17513 +
                                                                                     (res_17187 -
                                                                                      sext_i32_i64(chk_i_17511) *
                                                                                      hist_H_chk_17513)];
                            }
                            // execute operation
                            {
                                int64_t res_17183 = smax64(x_17181, x_17182);
                                
                                x_17181 = res_17183;
                            }
                            // update global result
                            {
                                ((volatile __local
                                  int64_t *) subhistogram_local_mem_17528)[sext_i32_i64(thread_local_subhisto_i_17533) *
                                                                           hist_H_chk_17513 +
                                                                           (res_17187 -
                                                                            sext_i32_i64(chk_i_17511) *
                                                                            hist_H_chk_17513)] =
                                    x_17181;
                            }
                            mem_fence_local();
                            old_17540 =
                                atomic_cmpxchg_i32_local(&((volatile __local
                                                            int *) locks_mem_17530)[sext_i32_i64(thread_local_subhisto_i_17533) *
                                                                                    hist_H_chk_17513 +
                                                                                    (res_17187 -
                                                                                     sext_i32_i64(chk_i_17511) *
                                                                                     hist_H_chk_17513)],
                                                         1, 0);
                            continue_17541 = 0;
                        }
                        mem_fence_local();
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Compact the multiple local memory subhistograms to result in global memory
        {
            int64_t trunc_H_17542 = smin64(hist_H_chk_17513, res_16524 -
                                           sext_i32_i64(chk_i_17511) *
                                           hist_H_chk_17513);
            int32_t histo_sizze_17543 = sext_i64_i32(trunc_H_17542);
            
            for (int32_t local_i_17544 = 0; local_i_17544 <
                 init_per_thread_17515; local_i_17544++) {
                int32_t j_17545 = local_i_17544 *
                        sext_i64_i32(max_group_sizze_17499) + local_tid_17517;
                
                if (slt32(j_17545, histo_sizze_17543)) {
                    int64_t x_17181;
                    int64_t x_17182;
                    
                    // Read values from subhistogram 0.
                    {
                        x_17181 = ((__local
                                    int64_t *) subhistogram_local_mem_17528)[sext_i32_i64(j_17545)];
                    }
                    // Accumulate based on values in other subhistograms.
                    {
                        for (int32_t subhisto_id_17546 = 0; subhisto_id_17546 <
                             hist_M_17506 - 1; subhisto_id_17546++) {
                            x_17182 = ((__local
                                        int64_t *) subhistogram_local_mem_17528)[(sext_i32_i64(subhisto_id_17546) +
                                                                                  1) *
                                                                                 hist_H_chk_17513 +
                                                                                 sext_i32_i64(j_17545)];
                            
                            int64_t res_17183;
                            
                            res_17183 = smax64(x_17181, x_17182);
                            x_17181 = res_17183;
                        }
                    }
                    // Put final bucket value in global memory.
                    {
                        ((__global
                          int64_t *) res_subhistos_mem_17490)[srem64(sext_i32_i64(virt_group_id_17523),
                                                                     num_groups_17500) *
                                                              res_16524 +
                                                              (sext_i32_i64(j_17545) +
                                                               sext_i32_i64(chk_i_17511) *
                                                               hist_H_chk_17513)] =
                            x_17181;
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
}
__kernel void mainzisegmap_16883(__global int *global_failure,
                                 int failure_is_an_option, __global
                                 int64_t *global_failure_args,
                                 int64_t paths_16408, int64_t steps_16409,
                                 float a_16413, float b_16414,
                                 float sigma_16415, float r0_16416,
                                 float dt_16420, int64_t upper_bound_16431,
                                 float res_16432, int64_t num_groups_17135,
                                 __global unsigned char *mem_17318, __global
                                 unsigned char *mem_17321, __global
                                 unsigned char *mem_17336)
{
    #define segmap_group_sizze_17134 (mainzisegmap_group_sizze_16885)
    
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
    
    int32_t global_tid_17415;
    int32_t local_tid_17416;
    int64_t group_sizze_17419;
    int32_t wave_sizze_17418;
    int32_t group_tid_17417;
    
    global_tid_17415 = get_global_id(0);
    local_tid_17416 = get_local_id(0);
    group_sizze_17419 = get_local_size(0);
    wave_sizze_17418 = LOCKSTEP_WIDTH;
    group_tid_17417 = get_group_id(0);
    
    int32_t phys_tid_16883;
    
    phys_tid_16883 = global_tid_17415;
    
    int32_t phys_group_id_17420;
    
    phys_group_id_17420 = get_group_id(0);
    for (int32_t i_17421 = 0; i_17421 <
         sdiv_up32(sext_i64_i32(sdiv_up64(paths_16408,
                                          segmap_group_sizze_17134)) -
                   phys_group_id_17420, sext_i64_i32(num_groups_17135));
         i_17421++) {
        int32_t virt_group_id_17422 = phys_group_id_17420 + i_17421 *
                sext_i64_i32(num_groups_17135);
        int64_t gtid_16882 = sext_i32_i64(virt_group_id_17422) *
                segmap_group_sizze_17134 + sext_i32_i64(local_tid_17416);
        
        if (slt64(gtid_16882, paths_16408)) {
            for (int64_t i_17423 = 0; i_17423 < steps_16409; i_17423++) {
                ((__global float *) mem_17321)[phys_tid_16883 + i_17423 *
                                               (num_groups_17135 *
                                                segmap_group_sizze_17134)] =
                    r0_16416;
            }
            for (int64_t i_17141 = 0; i_17141 < upper_bound_16431; i_17141++) {
                bool y_17143 = slt64(i_17141, steps_16409);
                bool index_certs_17144;
                
                if (!y_17143) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 0) ==
                            -1) {
                            global_failure_args[0] = i_17141;
                            global_failure_args[1] = steps_16409;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float shortstep_arg_17145 = ((__global
                                              float *) mem_17318)[i_17141 *
                                                                  paths_16408 +
                                                                  gtid_16882];
                float shortstep_arg_17146 = ((__global
                                              float *) mem_17321)[phys_tid_16883 +
                                                                  i_17141 *
                                                                  (num_groups_17135 *
                                                                   segmap_group_sizze_17134)];
                float y_17147 = b_16414 - shortstep_arg_17146;
                float x_17148 = a_16413 * y_17147;
                float x_17149 = dt_16420 * x_17148;
                float x_17150 = res_16432 * shortstep_arg_17145;
                float y_17151 = sigma_16415 * x_17150;
                float delta_r_17152 = x_17149 + y_17151;
                float res_17153 = shortstep_arg_17146 + delta_r_17152;
                int64_t i_17154 = add64(1, i_17141);
                bool x_17155 = sle64(0, i_17154);
                bool y_17156 = slt64(i_17154, steps_16409);
                bool bounds_check_17157 = x_17155 && y_17156;
                bool index_certs_17158;
                
                if (!bounds_check_17157) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 1) ==
                            -1) {
                            global_failure_args[0] = i_17154;
                            global_failure_args[1] = steps_16409;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                ((__global float *) mem_17321)[phys_tid_16883 + i_17154 *
                                               (num_groups_17135 *
                                                segmap_group_sizze_17134)] =
                    res_17153;
            }
            for (int64_t i_17425 = 0; i_17425 < steps_16409; i_17425++) {
                ((__global float *) mem_17336)[i_17425 * paths_16408 +
                                               gtid_16882] = ((__global
                                                               float *) mem_17321)[phys_tid_16883 +
                                                                                   i_17425 *
                                                                                   (num_groups_17135 *
                                                                                    segmap_group_sizze_17134)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_17134
}
__kernel void mainzisegmap_16981(__global int *global_failure,
                                 int64_t paths_16408, int64_t steps_16409,
                                 __global unsigned char *mem_17311, __global
                                 unsigned char *mem_17315)
{
    #define segmap_group_sizze_17089 (mainzisegmap_group_sizze_16984)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17409;
    int32_t local_tid_17410;
    int64_t group_sizze_17413;
    int32_t wave_sizze_17412;
    int32_t group_tid_17411;
    
    global_tid_17409 = get_global_id(0);
    local_tid_17410 = get_local_id(0);
    group_sizze_17413 = get_local_size(0);
    wave_sizze_17412 = LOCKSTEP_WIDTH;
    group_tid_17411 = get_group_id(0);
    
    int32_t phys_tid_16981;
    
    phys_tid_16981 = global_tid_17409;
    
    int64_t gtid_16979;
    
    gtid_16979 = squot64(sext_i32_i64(group_tid_17411) *
                         segmap_group_sizze_17089 +
                         sext_i32_i64(local_tid_17410), steps_16409);
    
    int64_t gtid_16980;
    
    gtid_16980 = sext_i32_i64(group_tid_17411) * segmap_group_sizze_17089 +
        sext_i32_i64(local_tid_17410) - squot64(sext_i32_i64(group_tid_17411) *
                                                segmap_group_sizze_17089 +
                                                sext_i32_i64(local_tid_17410),
                                                steps_16409) * steps_16409;
    if (slt64(gtid_16979, paths_16408) && slt64(gtid_16980, steps_16409)) {
        int32_t unsign_arg_17092 = ((__global int32_t *) mem_17311)[gtid_16979];
        int32_t res_17094 = sext_i64_i32(gtid_16980);
        int32_t x_17095 = lshr32(res_17094, 16);
        int32_t x_17096 = res_17094 ^ x_17095;
        int32_t x_17097 = mul32(73244475, x_17096);
        int32_t x_17098 = lshr32(x_17097, 16);
        int32_t x_17099 = x_17097 ^ x_17098;
        int32_t x_17100 = mul32(73244475, x_17099);
        int32_t x_17101 = lshr32(x_17100, 16);
        int32_t x_17102 = x_17100 ^ x_17101;
        int32_t unsign_arg_17103 = unsign_arg_17092 ^ x_17102;
        int32_t unsign_arg_17104 = mul32(48271, unsign_arg_17103);
        int32_t unsign_arg_17105 = umod32(unsign_arg_17104, 2147483647);
        int32_t unsign_arg_17106 = mul32(48271, unsign_arg_17105);
        int32_t unsign_arg_17107 = umod32(unsign_arg_17106, 2147483647);
        float res_17108 = uitofp_i32_f32(unsign_arg_17105);
        float res_17109 = res_17108 / 2.1474836e9F;
        float res_17110 = uitofp_i32_f32(unsign_arg_17107);
        float res_17111 = res_17110 / 2.1474836e9F;
        float res_17112;
        
        res_17112 = futrts_log32(res_17109);
        
        float res_17113 = -2.0F * res_17112;
        float res_17114;
        
        res_17114 = futrts_sqrt32(res_17113);
        
        float res_17115 = 6.2831855F * res_17111;
        float res_17116;
        
        res_17116 = futrts_cos32(res_17115);
        
        float res_17117 = res_17114 * res_17116;
        
        ((__global float *) mem_17315)[gtid_16979 * steps_16409 + gtid_16980] =
            res_17117;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_17089
}
__kernel void mainzisegmap_17045(__global int *global_failure,
                                 int64_t paths_16408, __global
                                 unsigned char *mem_17311)
{
    #define segmap_group_sizze_17064 (mainzisegmap_group_sizze_17047)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17404;
    int32_t local_tid_17405;
    int64_t group_sizze_17408;
    int32_t wave_sizze_17407;
    int32_t group_tid_17406;
    
    global_tid_17404 = get_global_id(0);
    local_tid_17405 = get_local_id(0);
    group_sizze_17408 = get_local_size(0);
    wave_sizze_17407 = LOCKSTEP_WIDTH;
    group_tid_17406 = get_group_id(0);
    
    int32_t phys_tid_17045;
    
    phys_tid_17045 = global_tid_17404;
    
    int64_t gtid_17044;
    
    gtid_17044 = sext_i32_i64(group_tid_17406) * segmap_group_sizze_17064 +
        sext_i32_i64(local_tid_17405);
    if (slt64(gtid_17044, paths_16408)) {
        int32_t res_17068 = sext_i64_i32(gtid_17044);
        int32_t x_17069 = lshr32(res_17068, 16);
        int32_t x_17070 = res_17068 ^ x_17069;
        int32_t x_17071 = mul32(73244475, x_17070);
        int32_t x_17072 = lshr32(x_17071, 16);
        int32_t x_17073 = x_17071 ^ x_17072;
        int32_t x_17074 = mul32(73244475, x_17073);
        int32_t x_17075 = lshr32(x_17074, 16);
        int32_t x_17076 = x_17074 ^ x_17075;
        int32_t unsign_arg_17077 = 777822902 ^ x_17076;
        int32_t unsign_arg_17078 = mul32(48271, unsign_arg_17077);
        int32_t unsign_arg_17079 = umod32(unsign_arg_17078, 2147483647);
        
        ((__global int32_t *) mem_17311)[gtid_17044] = unsign_arg_17079;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_17064
}
__kernel void mainzisegmap_17265(__global int *global_failure,
                                 int64_t res_16524, int64_t num_segments_16700,
                                 __global unsigned char *mem_17363, __global
                                 unsigned char *mem_17368, __global
                                 unsigned char *mem_17371, __global
                                 unsigned char *mem_17373)
{
    #define segmap_group_sizze_17268 (mainzisegmap_group_sizze_17267)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17897;
    int32_t local_tid_17898;
    int64_t group_sizze_17901;
    int32_t wave_sizze_17900;
    int32_t group_tid_17899;
    
    global_tid_17897 = get_global_id(0);
    local_tid_17898 = get_local_id(0);
    group_sizze_17901 = get_local_size(0);
    wave_sizze_17900 = LOCKSTEP_WIDTH;
    group_tid_17899 = get_group_id(0);
    
    int32_t phys_tid_17265;
    
    phys_tid_17265 = global_tid_17897;
    
    int64_t write_i_17264;
    
    write_i_17264 = sext_i32_i64(group_tid_17899) * segmap_group_sizze_17268 +
        sext_i32_i64(local_tid_17898);
    if (slt64(write_i_17264, res_16524)) {
        int64_t i_p_o_17293 = add64(1, write_i_17264);
        int64_t rot_i_17294 = smod64(i_p_o_17293, res_16524);
        bool x_16713 = ((__global bool *) mem_17363)[rot_i_17294];
        float write_value_16714 = ((__global float *) mem_17368)[write_i_17264];
        int64_t res_16715;
        
        if (x_16713) {
            int64_t x_16712 = ((__global int64_t *) mem_17371)[write_i_17264];
            int64_t res_16716 = sub64(x_16712, 1);
            
            res_16715 = res_16716;
        } else {
            res_16715 = -1;
        }
        if (sle64(0, res_16715) && slt64(res_16715, num_segments_16700)) {
            ((__global float *) mem_17373)[res_16715] = write_value_16714;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_17268
}
__kernel void mainzisegred_large_17575(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_17613_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_17611_backing_aligned_1,
                                       int64_t res_16524,
                                       int64_t num_groups_17177, __global
                                       unsigned char *mem_17350,
                                       int32_t num_subhistos_17489, __global
                                       unsigned char *res_subhistos_mem_17490,
                                       int64_t groups_per_segment_17597,
                                       int64_t elements_per_thread_17598,
                                       int64_t virt_num_groups_17599,
                                       int64_t threads_per_segment_17601,
                                       __global
                                       unsigned char *group_res_arr_mem_17602,
                                       __global
                                       unsigned char *mainzicounter_mem_17604)
{
    #define seghist_group_sizze_17175 (mainziseghist_group_sizze_17174)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_17613_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_17613_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_17611_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_17611_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17606;
    int32_t local_tid_17607;
    int64_t group_sizze_17610;
    int32_t wave_sizze_17609;
    int32_t group_tid_17608;
    
    global_tid_17606 = get_global_id(0);
    local_tid_17607 = get_local_id(0);
    group_sizze_17610 = get_local_size(0);
    wave_sizze_17609 = LOCKSTEP_WIDTH;
    group_tid_17608 = get_group_id(0);
    
    int32_t flat_gtid_17575;
    
    flat_gtid_17575 = global_tid_17606;
    
    __local char *red_arr_mem_17611;
    
    red_arr_mem_17611 = (__local char *) red_arr_mem_17611_backing_0;
    
    __local char *sync_arr_mem_17613;
    
    sync_arr_mem_17613 = (__local char *) sync_arr_mem_17613_backing_1;
    
    int32_t phys_group_id_17615;
    
    phys_group_id_17615 = get_group_id(0);
    for (int32_t i_17616 = 0; i_17616 <
         sdiv_up32(sext_i64_i32(virt_num_groups_17599) - phys_group_id_17615,
                   sext_i64_i32(num_groups_17177)); i_17616++) {
        int32_t virt_group_id_17617 = phys_group_id_17615 + i_17616 *
                sext_i64_i32(num_groups_17177);
        int32_t flat_segment_id_17618 = squot32(virt_group_id_17617,
                                                sext_i64_i32(groups_per_segment_17597));
        int64_t global_tid_17619 = srem64(sext_i32_i64(virt_group_id_17617) *
                                          seghist_group_sizze_17175 +
                                          sext_i32_i64(local_tid_17607),
                                          seghist_group_sizze_17175 *
                                          groups_per_segment_17597);
        int64_t bucket_id_17573 = sext_i32_i64(flat_segment_id_17618);
        int64_t subhistogram_id_17574;
        int64_t x_acc_17620;
        int64_t chunk_sizze_17621;
        
        chunk_sizze_17621 = smin64(elements_per_thread_17598,
                                   sdiv_up64(num_subhistos_17489 -
                                             sext_i32_i64(sext_i64_i32(global_tid_17619)),
                                             threads_per_segment_17601));
        
        int64_t x_17181;
        int64_t x_17182;
        
        // neutral-initialise the accumulators
        {
            x_acc_17620 = 0;
        }
        for (int64_t i_17625 = 0; i_17625 < chunk_sizze_17621; i_17625++) {
            subhistogram_id_17574 =
                sext_i32_i64(sext_i64_i32(global_tid_17619)) +
                threads_per_segment_17601 * i_17625;
            // apply map function
            {
                // load accumulator
                {
                    x_17181 = x_acc_17620;
                }
                // load new values
                {
                    x_17182 = ((__global
                                int64_t *) res_subhistos_mem_17490)[subhistogram_id_17574 *
                                                                    res_16524 +
                                                                    bucket_id_17573];
                }
                // apply reduction operator
                {
                    int64_t res_17183 = smax64(x_17181, x_17182);
                    
                    // store in accumulator
                    {
                        x_acc_17620 = res_17183;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_17181 = x_acc_17620;
            ((__local
              int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607)] =
                x_17181;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_17626;
        int32_t skip_waves_17627;
        
        skip_waves_17627 = 1;
        
        int64_t x_17622;
        int64_t x_17623;
        
        offset_17626 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_17607,
                      sext_i64_i32(seghist_group_sizze_17175))) {
                x_17622 = ((__local
                            int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607 +
                                                          offset_17626)];
            }
        }
        offset_17626 = 1;
        while (slt32(offset_17626, wave_sizze_17609)) {
            if (slt32(local_tid_17607 + offset_17626,
                      sext_i64_i32(seghist_group_sizze_17175)) &&
                ((local_tid_17607 - squot32(local_tid_17607, wave_sizze_17609) *
                  wave_sizze_17609) & (2 * offset_17626 - 1)) == 0) {
                // read array element
                {
                    x_17623 = ((volatile __local
                                int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607 +
                                                              offset_17626)];
                }
                // apply reduction operation
                {
                    int64_t res_17624 = smax64(x_17622, x_17623);
                    
                    x_17622 = res_17624;
                }
                // write result of operation
                {
                    ((volatile __local
                      int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607)] =
                        x_17622;
                }
            }
            offset_17626 *= 2;
        }
        while (slt32(skip_waves_17627,
                     squot32(sext_i64_i32(seghist_group_sizze_17175) +
                             wave_sizze_17609 - 1, wave_sizze_17609))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_17626 = skip_waves_17627 * wave_sizze_17609;
            if (slt32(local_tid_17607 + offset_17626,
                      sext_i64_i32(seghist_group_sizze_17175)) &&
                ((local_tid_17607 - squot32(local_tid_17607, wave_sizze_17609) *
                  wave_sizze_17609) == 0 && (squot32(local_tid_17607,
                                                     wave_sizze_17609) & (2 *
                                                                          skip_waves_17627 -
                                                                          1)) ==
                 0)) {
                // read array element
                {
                    x_17623 = ((__local
                                int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607 +
                                                              offset_17626)];
                }
                // apply reduction operation
                {
                    int64_t res_17624 = smax64(x_17622, x_17623);
                    
                    x_17622 = res_17624;
                }
                // write result of operation
                {
                    ((__local
                      int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607)] =
                        x_17622;
                }
            }
            skip_waves_17627 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (sext_i32_i64(local_tid_17607) == 0) {
                x_acc_17620 = x_17622;
            }
        }
        if (groups_per_segment_17597 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_17607 == 0) {
                    ((__global int64_t *) mem_17350)[bucket_id_17573] =
                        x_acc_17620;
                }
            }
        } else {
            int32_t old_counter_17628;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_17607 == 0) {
                    ((__global
                      int64_t *) group_res_arr_mem_17602)[sext_i32_i64(virt_group_id_17617) *
                                                          seghist_group_sizze_17175] =
                        x_acc_17620;
                    mem_fence_global();
                    old_counter_17628 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_17604)[sext_i32_i64(srem32(flat_segment_id_17618,
                                                                                                     10240))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_17613)[0] =
                        old_counter_17628 == groups_per_segment_17597 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_17629;
            
            is_last_group_17629 = ((__local bool *) sync_arr_mem_17613)[0];
            if (is_last_group_17629) {
                if (local_tid_17607 == 0) {
                    old_counter_17628 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_17604)[sext_i32_i64(srem32(flat_segment_id_17618,
                                                                                                     10240))],
                                              (int) (0 -
                                                     groups_per_segment_17597));
                }
                // read in the per-group-results
                {
                    int64_t read_per_thread_17630 =
                            sdiv_up64(groups_per_segment_17597,
                                      seghist_group_sizze_17175);
                    
                    x_17181 = 0;
                    for (int64_t i_17631 = 0; i_17631 < read_per_thread_17630;
                         i_17631++) {
                        int64_t group_res_id_17632 =
                                sext_i32_i64(local_tid_17607) *
                                read_per_thread_17630 + i_17631;
                        int64_t index_of_group_res_17633 =
                                sext_i32_i64(flat_segment_id_17618) *
                                groups_per_segment_17597 + group_res_id_17632;
                        
                        if (slt64(group_res_id_17632,
                                  groups_per_segment_17597)) {
                            x_17182 = ((__global
                                        int64_t *) group_res_arr_mem_17602)[index_of_group_res_17633 *
                                                                            seghist_group_sizze_17175];
                            
                            int64_t res_17183;
                            
                            res_17183 = smax64(x_17181, x_17182);
                            x_17181 = res_17183;
                        }
                    }
                }
                ((__local
                  int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607)] =
                    x_17181;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_17634;
                    int32_t skip_waves_17635;
                    
                    skip_waves_17635 = 1;
                    
                    int64_t x_17622;
                    int64_t x_17623;
                    
                    offset_17634 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_17607,
                                  sext_i64_i32(seghist_group_sizze_17175))) {
                            x_17622 = ((__local
                                        int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607 +
                                                                      offset_17634)];
                        }
                    }
                    offset_17634 = 1;
                    while (slt32(offset_17634, wave_sizze_17609)) {
                        if (slt32(local_tid_17607 + offset_17634,
                                  sext_i64_i32(seghist_group_sizze_17175)) &&
                            ((local_tid_17607 - squot32(local_tid_17607,
                                                        wave_sizze_17609) *
                              wave_sizze_17609) & (2 * offset_17634 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_17623 = ((volatile __local
                                            int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607 +
                                                                          offset_17634)];
                            }
                            // apply reduction operation
                            {
                                int64_t res_17624 = smax64(x_17622, x_17623);
                                
                                x_17622 = res_17624;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607)] =
                                    x_17622;
                            }
                        }
                        offset_17634 *= 2;
                    }
                    while (slt32(skip_waves_17635,
                                 squot32(sext_i64_i32(seghist_group_sizze_17175) +
                                         wave_sizze_17609 - 1,
                                         wave_sizze_17609))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_17634 = skip_waves_17635 * wave_sizze_17609;
                        if (slt32(local_tid_17607 + offset_17634,
                                  sext_i64_i32(seghist_group_sizze_17175)) &&
                            ((local_tid_17607 - squot32(local_tid_17607,
                                                        wave_sizze_17609) *
                              wave_sizze_17609) == 0 &&
                             (squot32(local_tid_17607, wave_sizze_17609) & (2 *
                                                                            skip_waves_17635 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_17623 = ((__local
                                            int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607 +
                                                                          offset_17634)];
                            }
                            // apply reduction operation
                            {
                                int64_t res_17624 = smax64(x_17622, x_17623);
                                
                                x_17622 = res_17624;
                            }
                            // write result of operation
                            {
                                ((__local
                                  int64_t *) red_arr_mem_17611)[sext_i32_i64(local_tid_17607)] =
                                    x_17622;
                            }
                        }
                        skip_waves_17635 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_17607 == 0) {
                            ((__global int64_t *) mem_17350)[bucket_id_17573] =
                                x_17622;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef seghist_group_sizze_17175
}
__kernel void mainzisegred_nonseg_17279(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_17915_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_17913_backing_aligned_1,
                                        int64_t num_segments_16700,
                                        int64_t num_groups_17274, __global
                                        unsigned char *mem_17373, __global
                                        unsigned char *mem_17377, __global
                                        unsigned char *mainzicounter_mem_17903,
                                        __global
                                        unsigned char *group_res_arr_mem_17905,
                                        int64_t num_threads_17907)
{
    #define segred_group_sizze_17272 (mainzisegred_group_sizze_17271)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_17915_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_17915_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_17913_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_17913_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17908;
    int32_t local_tid_17909;
    int64_t group_sizze_17912;
    int32_t wave_sizze_17911;
    int32_t group_tid_17910;
    
    global_tid_17908 = get_global_id(0);
    local_tid_17909 = get_local_id(0);
    group_sizze_17912 = get_local_size(0);
    wave_sizze_17911 = LOCKSTEP_WIDTH;
    group_tid_17910 = get_group_id(0);
    
    int32_t phys_tid_17279;
    
    phys_tid_17279 = global_tid_17908;
    
    __local char *sync_arr_mem_17913;
    
    sync_arr_mem_17913 = (__local char *) sync_arr_mem_17913_backing_0;
    
    __local char *red_arr_mem_17915;
    
    red_arr_mem_17915 = (__local char *) red_arr_mem_17915_backing_1;
    
    int64_t dummy_17277;
    
    dummy_17277 = 0;
    
    int64_t gtid_17278;
    
    gtid_17278 = 0;
    
    float x_acc_17917;
    int64_t chunk_sizze_17918;
    
    chunk_sizze_17918 = smin64(sdiv_up64(num_segments_16700,
                                         sext_i32_i64(sext_i64_i32(segred_group_sizze_17272 *
                                         num_groups_17274))),
                               sdiv_up64(num_segments_16700 -
                                         sext_i32_i64(phys_tid_17279),
                                         num_threads_17907));
    
    float x_16718;
    float x_16719;
    
    // neutral-initialise the accumulators
    {
        x_acc_17917 = 0.0F;
    }
    for (int64_t i_17922 = 0; i_17922 < chunk_sizze_17918; i_17922++) {
        gtid_17278 = sext_i32_i64(phys_tid_17279) + num_threads_17907 * i_17922;
        // apply map function
        {
            float x_16721 = ((__global float *) mem_17373)[gtid_17278];
            float res_16722 = fmax32(0.0F, x_16721);
            
            // save map-out results
            { }
            // load accumulator
            {
                x_16718 = x_acc_17917;
            }
            // load new values
            {
                x_16719 = res_16722;
            }
            // apply reduction operator
            {
                float res_16720 = x_16718 + x_16719;
                
                // store in accumulator
                {
                    x_acc_17917 = res_16720;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_16718 = x_acc_17917;
        ((__local float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909)] =
            x_16718;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_17923;
    int32_t skip_waves_17924;
    
    skip_waves_17924 = 1;
    
    float x_17919;
    float x_17920;
    
    offset_17923 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_17909, sext_i64_i32(segred_group_sizze_17272))) {
            x_17919 = ((__local
                        float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909 +
                                                    offset_17923)];
        }
    }
    offset_17923 = 1;
    while (slt32(offset_17923, wave_sizze_17911)) {
        if (slt32(local_tid_17909 + offset_17923,
                  sext_i64_i32(segred_group_sizze_17272)) && ((local_tid_17909 -
                                                               squot32(local_tid_17909,
                                                                       wave_sizze_17911) *
                                                               wave_sizze_17911) &
                                                              (2 *
                                                               offset_17923 -
                                                               1)) == 0) {
            // read array element
            {
                x_17920 = ((volatile __local
                            float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909 +
                                                        offset_17923)];
            }
            // apply reduction operation
            {
                float res_17921 = x_17919 + x_17920;
                
                x_17919 = res_17921;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909)] =
                    x_17919;
            }
        }
        offset_17923 *= 2;
    }
    while (slt32(skip_waves_17924,
                 squot32(sext_i64_i32(segred_group_sizze_17272) +
                         wave_sizze_17911 - 1, wave_sizze_17911))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_17923 = skip_waves_17924 * wave_sizze_17911;
        if (slt32(local_tid_17909 + offset_17923,
                  sext_i64_i32(segred_group_sizze_17272)) && ((local_tid_17909 -
                                                               squot32(local_tid_17909,
                                                                       wave_sizze_17911) *
                                                               wave_sizze_17911) ==
                                                              0 &&
                                                              (squot32(local_tid_17909,
                                                                       wave_sizze_17911) &
                                                               (2 *
                                                                skip_waves_17924 -
                                                                1)) == 0)) {
            // read array element
            {
                x_17920 = ((__local
                            float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909 +
                                                        offset_17923)];
            }
            // apply reduction operation
            {
                float res_17921 = x_17919 + x_17920;
                
                x_17919 = res_17921;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909)] =
                    x_17919;
            }
        }
        skip_waves_17924 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (sext_i32_i64(local_tid_17909) == 0) {
            x_acc_17917 = x_17919;
        }
    }
    
    int32_t old_counter_17925;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_17909 == 0) {
            ((__global
              float *) group_res_arr_mem_17905)[sext_i32_i64(group_tid_17910) *
                                                segred_group_sizze_17272] =
                x_acc_17917;
            mem_fence_global();
            old_counter_17925 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_17903)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_17913)[0] = old_counter_17925 ==
                num_groups_17274 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_17926;
    
    is_last_group_17926 = ((__local bool *) sync_arr_mem_17913)[0];
    if (is_last_group_17926) {
        if (local_tid_17909 == 0) {
            old_counter_17925 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_17903)[0],
                                                      (int) (0 -
                                                             num_groups_17274));
        }
        // read in the per-group-results
        {
            int64_t read_per_thread_17927 = sdiv_up64(num_groups_17274,
                                                      segred_group_sizze_17272);
            
            x_16718 = 0.0F;
            for (int64_t i_17928 = 0; i_17928 < read_per_thread_17927;
                 i_17928++) {
                int64_t group_res_id_17929 = sext_i32_i64(local_tid_17909) *
                        read_per_thread_17927 + i_17928;
                int64_t index_of_group_res_17930 = group_res_id_17929;
                
                if (slt64(group_res_id_17929, num_groups_17274)) {
                    x_16719 = ((__global
                                float *) group_res_arr_mem_17905)[index_of_group_res_17930 *
                                                                  segred_group_sizze_17272];
                    
                    float res_16720;
                    
                    res_16720 = x_16718 + x_16719;
                    x_16718 = res_16720;
                }
            }
        }
        ((__local float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909)] =
            x_16718;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_17931;
            int32_t skip_waves_17932;
            
            skip_waves_17932 = 1;
            
            float x_17919;
            float x_17920;
            
            offset_17931 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_17909,
                          sext_i64_i32(segred_group_sizze_17272))) {
                    x_17919 = ((__local
                                float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909 +
                                                            offset_17931)];
                }
            }
            offset_17931 = 1;
            while (slt32(offset_17931, wave_sizze_17911)) {
                if (slt32(local_tid_17909 + offset_17931,
                          sext_i64_i32(segred_group_sizze_17272)) &&
                    ((local_tid_17909 - squot32(local_tid_17909,
                                                wave_sizze_17911) *
                      wave_sizze_17911) & (2 * offset_17931 - 1)) == 0) {
                    // read array element
                    {
                        x_17920 = ((volatile __local
                                    float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909 +
                                                                offset_17931)];
                    }
                    // apply reduction operation
                    {
                        float res_17921 = x_17919 + x_17920;
                        
                        x_17919 = res_17921;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909)] =
                            x_17919;
                    }
                }
                offset_17931 *= 2;
            }
            while (slt32(skip_waves_17932,
                         squot32(sext_i64_i32(segred_group_sizze_17272) +
                                 wave_sizze_17911 - 1, wave_sizze_17911))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_17931 = skip_waves_17932 * wave_sizze_17911;
                if (slt32(local_tid_17909 + offset_17931,
                          sext_i64_i32(segred_group_sizze_17272)) &&
                    ((local_tid_17909 - squot32(local_tid_17909,
                                                wave_sizze_17911) *
                      wave_sizze_17911) == 0 && (squot32(local_tid_17909,
                                                         wave_sizze_17911) &
                                                 (2 * skip_waves_17932 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_17920 = ((__local
                                    float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909 +
                                                                offset_17931)];
                    }
                    // apply reduction operation
                    {
                        float res_17921 = x_17919 + x_17920;
                        
                        x_17919 = res_17921;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_17915)[sext_i32_i64(local_tid_17909)] =
                            x_17919;
                    }
                }
                skip_waves_17932 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_17909 == 0) {
                    ((__global float *) mem_17377)[0] = x_17919;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_17272
}
__kernel void mainzisegred_small_17575(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_17583_backing_aligned_0,
                                       int64_t res_16524,
                                       int64_t num_groups_17177, __global
                                       unsigned char *mem_17350,
                                       int32_t num_subhistos_17489, __global
                                       unsigned char *res_subhistos_mem_17490,
                                       int64_t segment_sizze_nonzzero_17576)
{
    #define seghist_group_sizze_17175 (mainziseghist_group_sizze_17174)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_17583_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_17583_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17578;
    int32_t local_tid_17579;
    int64_t group_sizze_17582;
    int32_t wave_sizze_17581;
    int32_t group_tid_17580;
    
    global_tid_17578 = get_global_id(0);
    local_tid_17579 = get_local_id(0);
    group_sizze_17582 = get_local_size(0);
    wave_sizze_17581 = LOCKSTEP_WIDTH;
    group_tid_17580 = get_group_id(0);
    
    int32_t flat_gtid_17575;
    
    flat_gtid_17575 = global_tid_17578;
    
    __local char *red_arr_mem_17583;
    
    red_arr_mem_17583 = (__local char *) red_arr_mem_17583_backing_0;
    
    int32_t phys_group_id_17585;
    
    phys_group_id_17585 = get_group_id(0);
    for (int32_t i_17586 = 0; i_17586 <
         sdiv_up32(sext_i64_i32(sdiv_up64(res_16524,
                                          squot64(seghist_group_sizze_17175,
                                                  segment_sizze_nonzzero_17576))) -
                   phys_group_id_17585, sext_i64_i32(num_groups_17177));
         i_17586++) {
        int32_t virt_group_id_17587 = phys_group_id_17585 + i_17586 *
                sext_i64_i32(num_groups_17177);
        int64_t bucket_id_17573 = squot64(sext_i32_i64(local_tid_17579),
                                          segment_sizze_nonzzero_17576) +
                sext_i32_i64(virt_group_id_17587) *
                squot64(seghist_group_sizze_17175,
                        segment_sizze_nonzzero_17576);
        int64_t subhistogram_id_17574 = srem64(sext_i32_i64(local_tid_17579),
                                               num_subhistos_17489);
        
        // apply map function if in bounds
        {
            if (slt64(0, num_subhistos_17489) && (slt64(bucket_id_17573,
                                                        res_16524) &&
                                                  slt64(sext_i32_i64(local_tid_17579),
                                                        num_subhistos_17489 *
                                                        squot64(seghist_group_sizze_17175,
                                                                segment_sizze_nonzzero_17576)))) {
                // save results to be reduced
                {
                    ((__local
                      int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)] =
                        ((__global
                          int64_t *) res_subhistos_mem_17490)[subhistogram_id_17574 *
                                                              res_16524 +
                                                              bucket_id_17573];
                }
            } else {
                ((__local
                  int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)] =
                    0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt64(0, num_subhistos_17489)) {
            // perform segmented scan to imitate reduction
            {
                int64_t x_17181;
                int64_t x_17182;
                int64_t x_17588;
                int64_t x_17589;
                bool ltid_in_bounds_17591;
                
                ltid_in_bounds_17591 = slt64(sext_i32_i64(local_tid_17579),
                                             num_subhistos_17489 *
                                             squot64(seghist_group_sizze_17175,
                                                     segment_sizze_nonzzero_17576));
                
                int32_t skip_threads_17592;
                
                // read input for in-block scan
                {
                    if (ltid_in_bounds_17591) {
                        x_17182 = ((volatile __local
                                    int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)];
                        if ((local_tid_17579 - squot32(local_tid_17579, 32) *
                             32) == 0) {
                            x_17181 = x_17182;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17592 = 1;
                    while (slt32(skip_threads_17592, 32)) {
                        if (sle32(skip_threads_17592, local_tid_17579 -
                                  squot32(local_tid_17579, 32) * 32) &&
                            ltid_in_bounds_17591) {
                            // read operands
                            {
                                x_17181 = ((volatile __local
                                            int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579) -
                                                                          sext_i32_i64(skip_threads_17592)];
                            }
                            // perform operation
                            {
                                bool inactive_17593 =
                                     slt64(srem64(sext_i32_i64(local_tid_17579),
                                                  num_subhistos_17489),
                                           sext_i32_i64(local_tid_17579) -
                                           sext_i32_i64(local_tid_17579 -
                                           skip_threads_17592));
                                
                                if (inactive_17593) {
                                    x_17181 = x_17182;
                                }
                                if (!inactive_17593) {
                                    int64_t res_17183 = smax64(x_17181,
                                                               x_17182);
                                    
                                    x_17181 = res_17183;
                                }
                            }
                        }
                        if (sle32(wave_sizze_17581, skip_threads_17592)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17592, local_tid_17579 -
                                  squot32(local_tid_17579, 32) * 32) &&
                            ltid_in_bounds_17591) {
                            // write result
                            {
                                ((volatile __local
                                  int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)] =
                                    x_17181;
                                x_17182 = x_17181;
                            }
                        }
                        if (sle32(wave_sizze_17581, skip_threads_17592)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17592 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_17579 - squot32(local_tid_17579, 32) * 32) ==
                        31 && ltid_in_bounds_17591) {
                        ((volatile __local
                          int64_t *) red_arr_mem_17583)[sext_i32_i64(squot32(local_tid_17579,
                                                                             32))] =
                            x_17181;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_17594;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_17579, 32) == 0 &&
                            ltid_in_bounds_17591) {
                            x_17589 = ((volatile __local
                                        int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)];
                            if ((local_tid_17579 - squot32(local_tid_17579,
                                                           32) * 32) == 0) {
                                x_17588 = x_17589;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_17594 = 1;
                        while (slt32(skip_threads_17594, 32)) {
                            if (sle32(skip_threads_17594, local_tid_17579 -
                                      squot32(local_tid_17579, 32) * 32) &&
                                (squot32(local_tid_17579, 32) == 0 &&
                                 ltid_in_bounds_17591)) {
                                // read operands
                                {
                                    x_17588 = ((volatile __local
                                                int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579) -
                                                                              sext_i32_i64(skip_threads_17594)];
                                }
                                // perform operation
                                {
                                    bool inactive_17595 =
                                         slt64(srem64(sext_i32_i64(local_tid_17579 *
                                                      32 + 32 - 1),
                                                      num_subhistos_17489),
                                               sext_i32_i64(local_tid_17579 *
                                               32 + 32 - 1) -
                                               sext_i32_i64((local_tid_17579 -
                                                             skip_threads_17594) *
                                               32 + 32 - 1));
                                    
                                    if (inactive_17595) {
                                        x_17588 = x_17589;
                                    }
                                    if (!inactive_17595) {
                                        int64_t res_17590 = smax64(x_17588,
                                                                   x_17589);
                                        
                                        x_17588 = res_17590;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_17581, skip_threads_17594)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_17594, local_tid_17579 -
                                      squot32(local_tid_17579, 32) * 32) &&
                                (squot32(local_tid_17579, 32) == 0 &&
                                 ltid_in_bounds_17591)) {
                                // write result
                                {
                                    ((volatile __local
                                      int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)] =
                                        x_17588;
                                    x_17589 = x_17588;
                                }
                            }
                            if (sle32(wave_sizze_17581, skip_threads_17594)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_17594 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_17579, 32) == 0 ||
                          !ltid_in_bounds_17591)) {
                        // read operands
                        {
                            x_17182 = x_17181;
                            x_17181 = ((__local
                                        int64_t *) red_arr_mem_17583)[sext_i32_i64(squot32(local_tid_17579,
                                                                                           32)) -
                                                                      1];
                        }
                        // perform operation
                        {
                            bool inactive_17596 =
                                 slt64(srem64(sext_i32_i64(local_tid_17579),
                                              num_subhistos_17489),
                                       sext_i32_i64(local_tid_17579) -
                                       sext_i32_i64(squot32(local_tid_17579,
                                                            32) * 32 - 1));
                            
                            if (inactive_17596) {
                                x_17181 = x_17182;
                            }
                            if (!inactive_17596) {
                                int64_t res_17183 = smax64(x_17181, x_17182);
                                
                                x_17181 = res_17183;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)] =
                                x_17181;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_17579, 32) == 0) {
                        ((__local
                          int64_t *) red_arr_mem_17583)[sext_i32_i64(local_tid_17579)] =
                            x_17182;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt64(sext_i32_i64(virt_group_id_17587) *
                      squot64(seghist_group_sizze_17175,
                              segment_sizze_nonzzero_17576) +
                      sext_i32_i64(local_tid_17579), res_16524) &&
                slt64(sext_i32_i64(local_tid_17579),
                      squot64(seghist_group_sizze_17175,
                              segment_sizze_nonzzero_17576))) {
                ((__global
                  int64_t *) mem_17350)[sext_i32_i64(virt_group_id_17587) *
                                        squot64(seghist_group_sizze_17175,
                                                segment_sizze_nonzzero_17576) +
                                        sext_i32_i64(local_tid_17579)] =
                    ((__local
                      int64_t *) red_arr_mem_17583)[(sext_i32_i64(local_tid_17579) +
                                                     1) *
                                                    segment_sizze_nonzzero_17576 -
                                                    1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef seghist_group_sizze_17175
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
    self.failure_msgs=["Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:84:112-119\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:105:1-145:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  cva.fut:84:58-120\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:105:1-145:18\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  cva.fut:135:36-88\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:133:18-138:46\n   #8  cva.fut:105:1-145:18\n"]
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
                                       all_sizes={"builtin#replicate_f32.group_size_17895": {"class": "group_size",
                                                                                   "value": None},
                                        "builtin#replicate_i64.group_size_17487": {"class": "group_size",
                                                                                   "value": None},
                                        "main.L2_size_17552": {"class": "L2_for_histogram", "value": 4194304},
                                        "main.seghist_group_size_17174": {"class": "group_size", "value": None},
                                        "main.seghist_num_groups_17176": {"class": "num_groups", "value": None},
                                        "main.segmap_group_size_16885": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_16984": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_17047": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_17267": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_16887": {"class": "num_groups", "value": None},
                                        "main.segred_group_size_17271": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_17273": {"class": "num_groups", "value": None},
                                        "main.segscan_group_size_17166": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_17189": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_17197": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_17205": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_17257": {"class": "group_size", "value": None},
                                        "main.segscan_num_groups_17168": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_17191": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_17199": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_17207": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_17259": {"class": "num_groups", "value": None}})
    self.builtinzhreplicate_f32zireplicate_17892_var = program.builtinzhreplicate_f32zireplicate_17892
    self.builtinzhreplicate_i64zireplicate_17484_var = program.builtinzhreplicate_i64zireplicate_17484
    self.gpu_map_transpose_f32_var = program.gpu_map_transpose_f32
    self.gpu_map_transpose_f32_low_height_var = program.gpu_map_transpose_f32_low_height
    self.gpu_map_transpose_f32_low_width_var = program.gpu_map_transpose_f32_low_width
    self.gpu_map_transpose_f32_small_var = program.gpu_map_transpose_f32_small
    self.mainziscan_stage1_17172_var = program.mainziscan_stage1_17172
    self.mainziscan_stage1_17195_var = program.mainziscan_stage1_17195
    self.mainziscan_stage1_17203_var = program.mainziscan_stage1_17203
    self.mainziscan_stage1_17211_var = program.mainziscan_stage1_17211
    self.mainziscan_stage1_17263_var = program.mainziscan_stage1_17263
    self.mainziscan_stage2_17172_var = program.mainziscan_stage2_17172
    self.mainziscan_stage2_17195_var = program.mainziscan_stage2_17195
    self.mainziscan_stage2_17203_var = program.mainziscan_stage2_17203
    self.mainziscan_stage2_17211_var = program.mainziscan_stage2_17211
    self.mainziscan_stage2_17263_var = program.mainziscan_stage2_17263
    self.mainziscan_stage3_17172_var = program.mainziscan_stage3_17172
    self.mainziscan_stage3_17195_var = program.mainziscan_stage3_17195
    self.mainziscan_stage3_17203_var = program.mainziscan_stage3_17203
    self.mainziscan_stage3_17211_var = program.mainziscan_stage3_17211
    self.mainziscan_stage3_17263_var = program.mainziscan_stage3_17263
    self.mainziseghist_global_17180_var = program.mainziseghist_global_17180
    self.mainziseghist_local_17180_var = program.mainziseghist_local_17180
    self.mainzisegmap_16883_var = program.mainzisegmap_16883
    self.mainzisegmap_16981_var = program.mainzisegmap_16981
    self.mainzisegmap_17045_var = program.mainzisegmap_17045
    self.mainzisegmap_17265_var = program.mainzisegmap_17265
    self.mainzisegred_large_17575_var = program.mainzisegred_large_17575
    self.mainzisegred_nonseg_17279_var = program.mainzisegred_nonseg_17279
    self.mainzisegred_small_17575_var = program.mainzisegred_small_17575
    self.constants = {}
    mainzihist_locks_mem_17560 = np.zeros(100151, dtype=np.int32)
    static_mem_17935 = opencl_alloc(self, 400604, "static_mem_17935")
    if (400604 != 0):
      cl.enqueue_copy(self.queue, static_mem_17935,
                      normaliseArray(mainzihist_locks_mem_17560),
                      is_blocking=synchronous)
    self.mainzihist_locks_mem_17560 = static_mem_17935
    mainzicounter_mem_17604 = np.zeros(10240, dtype=np.int32)
    static_mem_17938 = opencl_alloc(self, 40960, "static_mem_17938")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_17938,
                      normaliseArray(mainzicounter_mem_17604),
                      is_blocking=synchronous)
    self.mainzicounter_mem_17604 = static_mem_17938
    mainzicounter_mem_17903 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_17940 = opencl_alloc(self, 40, "static_mem_17940")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_17940,
                      normaliseArray(mainzicounter_mem_17903),
                      is_blocking=synchronous)
    self.mainzicounter_mem_17903 = static_mem_17940
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
  def futhark_builtinzhreplicate_f32(self, mem_17888, num_elems_17889,
                                     val_17890):
    group_sizze_17895 = self.sizes["builtin#replicate_f32.group_size_17895"]
    num_groups_17896 = sdiv_up64(num_elems_17889, group_sizze_17895)
    if ((1 * (np.long(num_groups_17896) * np.long(group_sizze_17895))) != 0):
      self.builtinzhreplicate_f32zireplicate_17892_var.set_args(mem_17888,
                                                                np.int32(num_elems_17889),
                                                                np.float32(val_17890))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_f32zireplicate_17892_var,
                                 ((np.long(num_groups_17896) * np.long(group_sizze_17895)),),
                                 (np.long(group_sizze_17895),))
      if synchronous:
        sync(self)
    return ()
  def futhark_builtinzhreplicate_i64(self, mem_17480, num_elems_17481,
                                     val_17482):
    group_sizze_17487 = self.sizes["builtin#replicate_i64.group_size_17487"]
    num_groups_17488 = sdiv_up64(num_elems_17481, group_sizze_17487)
    if ((1 * (np.long(num_groups_17488) * np.long(group_sizze_17487))) != 0):
      self.builtinzhreplicate_i64zireplicate_17484_var.set_args(mem_17480,
                                                                np.int32(num_elems_17481),
                                                                np.int64(val_17482))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_i64zireplicate_17484_var,
                                 ((np.long(num_groups_17488) * np.long(group_sizze_17487)),),
                                 (np.long(group_sizze_17487),))
      if synchronous:
        sync(self)
    return ()
  def futhark_main(self, paths_16408, steps_16409, swap_term_16410,
                   payments_16411, notional_16412, a_16413, b_16414,
                   sigma_16415, r0_16416):
    res_16417 = sitofp_i64_f32(payments_16411)
    x_16418 = (swap_term_16410 * res_16417)
    res_16419 = sitofp_i64_f32(steps_16409)
    dt_16420 = (x_16418 / res_16419)
    sims_per_year_16421 = (res_16419 / x_16418)
    bounds_invalid_upwards_16422 = slt64(steps_16409, np.int64(1))
    valid_16423 = not(bounds_invalid_upwards_16422)
    range_valid_c_16424 = True
    assert valid_16423, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  cva.fut:75:56-67\n   #1  cva.fut:112:17-40\n   #2  cva.fut:105:1-145:18\n" % ("Range ",
                                                                                                                                                    np.int64(1),
                                                                                                                                                    "..",
                                                                                                                                                    np.int64(2),
                                                                                                                                                    "...",
                                                                                                                                                    steps_16409,
                                                                                                                                                    " is invalid."))
    bounds_invalid_upwards_16426 = slt64(paths_16408, np.int64(0))
    valid_16427 = not(bounds_invalid_upwards_16426)
    range_valid_c_16428 = True
    assert valid_16427, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:113:19-49\n   #4  cva.fut:105:1-145:18\n" % ("Range ",
                                                                                                                                                                                                                                                                np.int64(0),
                                                                                                                                                                                                                                                                "..",
                                                                                                                                                                                                                                                                np.int64(1),
                                                                                                                                                                                                                                                                "..<",
                                                                                                                                                                                                                                                                paths_16408,
                                                                                                                                                                                                                                                                " is invalid."))
    upper_bound_16431 = (steps_16409 - np.int64(1))
    res_16432 = futhark_sqrt32(dt_16420)
    segmap_group_sizze_17064 = self.sizes["main.segmap_group_size_17047"]
    segmap_usable_groups_17065 = sdiv_up64(paths_16408,
                                           segmap_group_sizze_17064)
    bytes_17310 = (np.int64(4) * paths_16408)
    mem_17311 = opencl_alloc(self, bytes_17310, "mem_17311")
    if ((1 * (np.long(segmap_usable_groups_17065) * np.long(segmap_group_sizze_17064))) != 0):
      self.mainzisegmap_17045_var.set_args(self.global_failure,
                                           np.int64(paths_16408), mem_17311)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_17045_var,
                                 ((np.long(segmap_usable_groups_17065) * np.long(segmap_group_sizze_17064)),),
                                 (np.long(segmap_group_sizze_17064),))
      if synchronous:
        sync(self)
    nest_sizze_17088 = (paths_16408 * steps_16409)
    segmap_group_sizze_17089 = self.sizes["main.segmap_group_size_16984"]
    segmap_usable_groups_17090 = sdiv_up64(nest_sizze_17088,
                                           segmap_group_sizze_17089)
    bytes_17313 = (np.int64(4) * nest_sizze_17088)
    mem_17315 = opencl_alloc(self, bytes_17313, "mem_17315")
    if ((1 * (np.long(segmap_usable_groups_17090) * np.long(segmap_group_sizze_17089))) != 0):
      self.mainzisegmap_16981_var.set_args(self.global_failure,
                                           np.int64(paths_16408),
                                           np.int64(steps_16409), mem_17311,
                                           mem_17315)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_16981_var,
                                 ((np.long(segmap_usable_groups_17090) * np.long(segmap_group_sizze_17089)),),
                                 (np.long(segmap_group_sizze_17089),))
      if synchronous:
        sync(self)
    mem_17311 = None
    segmap_group_sizze_17134 = self.sizes["main.segmap_group_size_16885"]
    max_num_groups_17414 = self.sizes["main.segmap_num_groups_16887"]
    num_groups_17135 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_16408,
                                                            sext_i32_i64(segmap_group_sizze_17134)),
                                                  sext_i32_i64(max_num_groups_17414))))
    mem_17318 = opencl_alloc(self, bytes_17313, "mem_17318")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_17318, np.int64(0),
                                                mem_17315, np.int64(0),
                                                np.int64(1), steps_16409,
                                                paths_16408)
    mem_17315 = None
    mem_17336 = opencl_alloc(self, bytes_17313, "mem_17336")
    bytes_17320 = (np.int64(4) * steps_16409)
    num_threads_17391 = (segmap_group_sizze_17134 * num_groups_17135)
    total_sizze_17392 = (bytes_17320 * num_threads_17391)
    mem_17321 = opencl_alloc(self, total_sizze_17392, "mem_17321")
    if ((1 * (np.long(num_groups_17135) * np.long(segmap_group_sizze_17134))) != 0):
      self.mainzisegmap_16883_var.set_args(self.global_failure,
                                           self.failure_is_an_option,
                                           self.global_failure_args,
                                           np.int64(paths_16408),
                                           np.int64(steps_16409),
                                           np.float32(a_16413),
                                           np.float32(b_16414),
                                           np.float32(sigma_16415),
                                           np.float32(r0_16416),
                                           np.float32(dt_16420),
                                           np.int64(upper_bound_16431),
                                           np.float32(res_16432),
                                           np.int64(num_groups_17135),
                                           mem_17318, mem_17321, mem_17336)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_16883_var,
                                 ((np.long(num_groups_17135) * np.long(segmap_group_sizze_17134)),),
                                 (np.long(segmap_group_sizze_17134),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mem_17318 = None
    mem_17321 = None
    fold_input_is_empty_16494 = (paths_16408 == np.int64(0))
    res_16495 = sitofp_i64_f32(paths_16408)
    x_16496 = fpow32(a_16413, np.float32(2.0))
    x_16497 = (b_16414 * x_16496)
    x_16498 = fpow32(sigma_16415, np.float32(2.0))
    y_16499 = (x_16498 / np.float32(2.0))
    y_16500 = (x_16497 - y_16499)
    y_16501 = (np.float32(4.0) * a_16413)
    mem_17338 = opencl_alloc(self, bytes_17320, "mem_17338")
    segscan_group_sizze_17167 = self.sizes["main.segscan_group_size_17166"]
    max_num_groups_17426 = self.sizes["main.segscan_num_groups_17168"]
    num_groups_17169 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_16408,
                                                            sext_i32_i64(segscan_group_sizze_17167)),
                                                  sext_i32_i64(max_num_groups_17426))))
    seghist_group_sizze_17175 = self.sizes["main.seghist_group_size_17174"]
    max_num_groups_17427 = self.sizes["main.seghist_num_groups_17176"]
    num_groups_17177 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(paths_16408,
                                                            sext_i32_i64(seghist_group_sizze_17175)),
                                                  sext_i32_i64(max_num_groups_17427))))
    segscan_group_sizze_17190 = self.sizes["main.segscan_group_size_17189"]
    segscan_group_sizze_17198 = self.sizes["main.segscan_group_size_17197"]
    segscan_group_sizze_17206 = self.sizes["main.segscan_group_size_17205"]
    segscan_group_sizze_17258 = self.sizes["main.segscan_group_size_17257"]
    segred_group_sizze_17272 = self.sizes["main.segred_group_size_17271"]
    bytes_17347 = (np.int64(8) * paths_16408)
    mem_17348 = opencl_alloc(self, bytes_17347, "mem_17348")
    mem_17377 = opencl_alloc(self, np.int64(4), "mem_17377")
    redout_17161 = np.float32(0.0)
    i_17163 = np.int64(0)
    one_17943 = np.int64(1)
    for counter_17942 in range(steps_16409):
      index_primexp_17296 = (np.int64(1) + i_17163)
      res_16510 = sitofp_i64_f32(index_primexp_17296)
      res_16511 = (res_16510 / sims_per_year_16421)
      x_16512 = (res_16511 / swap_term_16410)
      ceil_arg_16513 = (x_16512 - np.float32(1.0))
      res_16514 = futhark_ceil32(ceil_arg_16513)
      res_16515 = fptosi_f32_i64(res_16514)
      res_16516 = (payments_16411 - res_16515)
      if slt64(np.int64(0), paths_16408):
        stage1_max_num_groups_17430 = self.max_group_size
        stage1_num_groups_17431 = smin64(stage1_max_num_groups_17430,
                                         num_groups_17169)
        num_threads_17432 = sext_i64_i32((stage1_num_groups_17431 * segscan_group_sizze_17167))
        if ((1 * (np.long(stage1_num_groups_17431) * np.long(segscan_group_sizze_17167))) != 0):
          self.mainziscan_stage1_17172_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_17167)))),
                                                    np.int64(paths_16408),
                                                    np.int64(res_16516),
                                                    mem_17348,
                                                    np.int32(num_threads_17432))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_17172_var,
                                     ((np.long(stage1_num_groups_17431) * np.long(segscan_group_sizze_17167)),),
                                     (np.long(segscan_group_sizze_17167),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_17431))) != 0):
          self.mainziscan_stage2_17172_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_17431)))),
                                                    np.int64(paths_16408),
                                                    mem_17348,
                                                    np.int64(stage1_num_groups_17431),
                                                    np.int32(num_threads_17432))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_17172_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_17431)),),
                                     (np.long(stage1_num_groups_17431),))
          if synchronous:
            sync(self)
        required_groups_17468 = sext_i64_i32(sdiv_up64(paths_16408,
                                                       segscan_group_sizze_17167))
        if ((1 * (np.long(num_groups_17169) * np.long(segscan_group_sizze_17167))) != 0):
          self.mainziscan_stage3_17172_var.set_args(self.global_failure,
                                                    np.int64(paths_16408),
                                                    np.int64(num_groups_17169),
                                                    mem_17348,
                                                    np.int32(num_threads_17432),
                                                    np.int32(required_groups_17468))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_17172_var,
                                     ((np.long(num_groups_17169) * np.long(segscan_group_sizze_17167)),),
                                     (np.long(segscan_group_sizze_17167),))
          if synchronous:
            sync(self)
      if fold_input_is_empty_16494:
        res_16524 = np.int64(0)
      else:
        y_16525 = (paths_16408 * res_16516)
        res_16524 = y_16525
      bounds_invalid_upwards_16526 = slt64(res_16524, np.int64(0))
      valid_16527 = not(bounds_invalid_upwards_16526)
      range_valid_c_16528 = True
      assert valid_16527, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  cva.fut:135:36-88\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:133:18-138:46\n   #8  cva.fut:105:1-145:18\n" % ("Range ",
                                                                                                                                                                                                                                                                                                                                                                                                                                         np.int64(0),
                                                                                                                                                                                                                                                                                                                                                                                                                                         "..",
                                                                                                                                                                                                                                                                                                                                                                                                                                         np.int64(1),
                                                                                                                                                                                                                                                                                                                                                                                                                                         "..<",
                                                                                                                                                                                                                                                                                                                                                                                                                                         res_16524,
                                                                                                                                                                                                                                                                                                                                                                                                                                         " is invalid."))
      bytes_17349 = (np.int64(8) * res_16524)
      mem_17350 = opencl_alloc(self, bytes_17349, "mem_17350")
      self.futhark_builtinzhreplicate_i64(mem_17350, res_16524, np.int64(0))
      h_17492 = (np.int32(8) * res_16524)
      seg_h_17493 = (np.int32(8) * res_16524)
      if (seg_h_17493 == np.int64(0)):
        pass
      else:
        hist_H_17494 = res_16524
        hist_el_sizze_17495 = (sdiv_up64(h_17492, hist_H_17494) + np.int64(4))
        hist_N_17496 = paths_16408
        hist_RF_17497 = np.int64(1)
        hist_L_17498 = self.max_local_memory
        max_group_sizze_17499 = self.max_group_size
        num_groups_17500 = sdiv_up64(sext_i32_i64(sext_i64_i32((num_groups_17177 * seghist_group_sizze_17175))),
                                     max_group_sizze_17499)
        hist_m_prime_17501 = (sitofp_i64_f64(smin64(sext_i32_i64(squot32(hist_L_17498,
                                                                         hist_el_sizze_17495)),
                                                    sdiv_up64(hist_N_17496,
                                                              num_groups_17500))) / sitofp_i64_f64(hist_H_17494))
        hist_M0_17502 = smax64(np.int64(1),
                               smin64(fptosi_f64_i64(hist_m_prime_17501),
                                      max_group_sizze_17499))
        hist_Nout_17503 = np.int64(1)
        hist_Nin_17504 = paths_16408
        work_asymp_M_max_17505 = squot64((hist_Nout_17503 * hist_N_17496),
                                         ((np.int64(2) * num_groups_17500) * hist_H_17494))
        hist_M_17506 = sext_i64_i32(smin64(hist_M0_17502,
                                           work_asymp_M_max_17505))
        hist_C_17507 = sdiv_up64(max_group_sizze_17499,
                                 sext_i32_i64(smax32(np.int32(1),
                                                     hist_M_17506)))
        local_mem_needed_17508 = (hist_el_sizze_17495 * sext_i32_i64(hist_M_17506))
        hist_S_17509 = sext_i64_i32(sdiv_up64((hist_H_17494 * local_mem_needed_17508),
                                              hist_L_17498))
        if (sle64(hist_H_17494,
                  hist_Nin_17504) and (sle64(local_mem_needed_17508,
                                             hist_L_17498) and (sle32(hist_S_17509,
                                                                      np.int32(6)) and (sle64(hist_C_17507,
                                                                                              max_group_sizze_17499) and slt32(np.int32(0),
                                                                                                                               hist_M_17506))))):
          num_segments_17510 = np.int64(1)
          num_subhistos_17489 = (num_groups_17500 * num_segments_17510)
          if (num_subhistos_17489 == np.int64(1)):
            res_subhistos_mem_17490 = mem_17350
          else:
            res_subhistos_mem_17490 = opencl_alloc(self,
                                                   ((sext_i32_i64(num_subhistos_17489) * res_16524) * np.int32(8)),
                                                   "res_subhistos_mem_17490")
            self.futhark_builtinzhreplicate_i64(res_subhistos_mem_17490,
                                                (num_subhistos_17489 * res_16524),
                                                np.int64(0))
            if ((res_16524 * np.int32(8)) != 0):
              cl.enqueue_copy(self.queue, res_subhistos_mem_17490, mem_17350,
                              dest_offset=np.long(np.int64(0)),
                              src_offset=np.long(np.int64(0)),
                              byte_count=np.long((res_16524 * np.int32(8))))
            if synchronous:
              sync(self)
          chk_i_17511 = np.int32(0)
          one_17934 = np.int32(1)
          for counter_17933 in range(hist_S_17509):
            num_segments_17512 = np.int64(1)
            hist_H_chk_17513 = sdiv_up64(res_16524, sext_i32_i64(hist_S_17509))
            histo_sizze_17514 = hist_H_chk_17513
            init_per_thread_17515 = sext_i64_i32(sdiv_up64((sext_i32_i64(hist_M_17506) * histo_sizze_17514),
                                                           max_group_sizze_17499))
            if ((1 * (np.long(num_groups_17500) * np.long(max_group_sizze_17499))) != 0):
              self.mainziseghist_local_17180_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long((np.int32(4) * (hist_M_17506 * hist_H_chk_17513)))),
                                                          cl.LocalMemory(np.long((np.int32(8) * (hist_M_17506 * hist_H_chk_17513)))),
                                                          np.int64(paths_16408),
                                                          np.int64(res_16524),
                                                          mem_17348,
                                                          res_subhistos_mem_17490,
                                                          np.int32(max_group_sizze_17499),
                                                          np.int64(num_groups_17500),
                                                          np.int32(hist_M_17506),
                                                          np.int32(chk_i_17511),
                                                          np.int64(num_segments_17512),
                                                          np.int64(hist_H_chk_17513),
                                                          np.int64(histo_sizze_17514),
                                                          np.int32(init_per_thread_17515))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainziseghist_local_17180_var,
                                         ((np.long(num_groups_17500) * np.long(max_group_sizze_17499)),),
                                         (np.long(max_group_sizze_17499),))
              if synchronous:
                sync(self)
            chk_i_17511 += one_17934
        else:
          hist_H_17547 = res_16524
          hist_RF_17548 = ((np.float64(0.0) + sitofp_i32_f64(np.int64(1))) / np.float64(1.0))
          hist_el_sizze_17549 = squot32(sext_i64_i32((np.int32(4) + np.int32(8))),
                                        np.int32(2))
          hist_C_max_17550 = fmin64(sitofp_i32_f64(sext_i64_i32((num_groups_17177 * seghist_group_sizze_17175))),
                                    (sitofp_i32_f64(hist_H_17547) / np.float64(2.0)))
          hist_M_min_17551 = smax32(np.int32(1),
                                    sext_i64_i32(fptosi_f64_i64((sitofp_i32_f64(sext_i64_i32((num_groups_17177 * seghist_group_sizze_17175))) / hist_C_max_17550))))
          L2_sizze_17552 = self.sizes["main.L2_size_17552"]
          hist_RACE_exp_17553 = fmax64(np.float64(1.0),
                                       ((np.float64(0.75) * hist_RF_17548) / (np.float64(64.0) / sitofp_i32_f64(hist_el_sizze_17549))))
          if slt64(paths_16408, hist_H_17547):
            hist_S_17554 = np.int32(1)
          else:
            hist_S_17554 = sext_i64_i32(sdiv_up64(((sext_i32_i64(hist_M_min_17551) * hist_H_17547) * sext_i32_i64(hist_el_sizze_17549)),
                                                  fptosi_f64_i64(((np.float64(0.4) * sitofp_i32_f64(L2_sizze_17552)) * hist_RACE_exp_17553))))
          hist_H_chk_17555 = sdiv_up64(res_16524, sext_i32_i64(hist_S_17554))
          hist_k_max_17556 = (fmin64(((np.float64(0.4) * (sitofp_i32_f64(L2_sizze_17552) / sitofp_i32_f64(sext_i64_i32((np.int32(4) + np.int32(8)))))) * hist_RACE_exp_17553),
                                     sitofp_i32_f64(paths_16408)) / sitofp_i32_f64(sext_i64_i32((num_groups_17177 * seghist_group_sizze_17175))))
          hist_u_17557 = np.int64(1)
          hist_C_17558 = fmin64(sitofp_i32_f64(sext_i64_i32((num_groups_17177 * seghist_group_sizze_17175))),
                                (sitofp_i32_f64((hist_u_17557 * hist_H_chk_17555)) / hist_k_max_17556))
          hist_M_17559 = smax32(hist_M_min_17551,
                                sext_i64_i32(fptosi_f64_i64((sitofp_i32_f64(sext_i64_i32((num_groups_17177 * seghist_group_sizze_17175))) / hist_C_17558))))
          num_subhistos_17489 = sext_i32_i64(hist_M_17559)
          if (hist_M_17559 == np.int32(1)):
            res_subhistos_mem_17490 = mem_17350
          else:
            if (num_subhistos_17489 == np.int64(1)):
              res_subhistos_mem_17490 = mem_17350
            else:
              res_subhistos_mem_17490 = opencl_alloc(self,
                                                     ((sext_i32_i64(num_subhistos_17489) * res_16524) * np.int32(8)),
                                                     "res_subhistos_mem_17490")
              self.futhark_builtinzhreplicate_i64(res_subhistos_mem_17490,
                                                  (num_subhistos_17489 * res_16524),
                                                  np.int64(0))
              if ((res_16524 * np.int32(8)) != 0):
                cl.enqueue_copy(self.queue, res_subhistos_mem_17490, mem_17350,
                                dest_offset=np.long(np.int64(0)),
                                src_offset=np.long(np.int64(0)),
                                byte_count=np.long((res_16524 * np.int32(8))))
              if synchronous:
                sync(self)
          mainzihist_locks_mem_17560 = self.mainzihist_locks_mem_17560
          chk_i_17562 = np.int32(0)
          one_17937 = np.int32(1)
          for counter_17936 in range(hist_S_17554):
            hist_H_chk_17563 = sdiv_up64(res_16524, sext_i32_i64(hist_S_17554))
            if ((1 * (np.long(num_groups_17177) * np.long(seghist_group_sizze_17175))) != 0):
              self.mainziseghist_global_17180_var.set_args(self.global_failure,
                                                           np.int64(paths_16408),
                                                           np.int64(res_16524),
                                                           np.int64(num_groups_17177),
                                                           mem_17348,
                                                           np.int32(num_subhistos_17489),
                                                           res_subhistos_mem_17490,
                                                           mainzihist_locks_mem_17560,
                                                           np.int32(chk_i_17562),
                                                           np.int64(hist_H_chk_17563))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainziseghist_global_17180_var,
                                         ((np.long(num_groups_17177) * np.long(seghist_group_sizze_17175)),),
                                         (np.long(seghist_group_sizze_17175),))
              if synchronous:
                sync(self)
            chk_i_17562 += one_17937
        if (num_subhistos_17489 == np.int64(1)):
          mem_17350 = res_subhistos_mem_17490
        else:
          if slt64((num_subhistos_17489 * np.int64(2)),
                   seghist_group_sizze_17175):
            segment_sizze_nonzzero_17576 = smax64(np.int64(1),
                                                  num_subhistos_17489)
            num_threads_17577 = (num_groups_17177 * seghist_group_sizze_17175)
            if ((1 * (np.long(num_groups_17177) * np.long(seghist_group_sizze_17175))) != 0):
              self.mainzisegred_small_17575_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long((np.int32(8) * seghist_group_sizze_17175))),
                                                         np.int64(res_16524),
                                                         np.int64(num_groups_17177),
                                                         mem_17350,
                                                         np.int32(num_subhistos_17489),
                                                         res_subhistos_mem_17490,
                                                         np.int64(segment_sizze_nonzzero_17576))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_small_17575_var,
                                         ((np.long(num_groups_17177) * np.long(seghist_group_sizze_17175)),),
                                         (np.long(seghist_group_sizze_17175),))
              if synchronous:
                sync(self)
          else:
            groups_per_segment_17597 = sdiv_up64(num_groups_17177,
                                                 smax64(np.int64(1), res_16524))
            elements_per_thread_17598 = sdiv_up64(num_subhistos_17489,
                                                  (seghist_group_sizze_17175 * groups_per_segment_17597))
            virt_num_groups_17599 = (groups_per_segment_17597 * res_16524)
            num_threads_17600 = (num_groups_17177 * seghist_group_sizze_17175)
            threads_per_segment_17601 = (groups_per_segment_17597 * seghist_group_sizze_17175)
            group_res_arr_mem_17602 = opencl_alloc(self,
                                                   (np.int32(8) * (seghist_group_sizze_17175 * virt_num_groups_17599)),
                                                   "group_res_arr_mem_17602")
            mainzicounter_mem_17604 = self.mainzicounter_mem_17604
            if ((1 * (np.long(num_groups_17177) * np.long(seghist_group_sizze_17175))) != 0):
              self.mainzisegred_large_17575_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long(np.int32(1))),
                                                         cl.LocalMemory(np.long((np.int32(8) * seghist_group_sizze_17175))),
                                                         np.int64(res_16524),
                                                         np.int64(num_groups_17177),
                                                         mem_17350,
                                                         np.int32(num_subhistos_17489),
                                                         res_subhistos_mem_17490,
                                                         np.int64(groups_per_segment_17597),
                                                         np.int64(elements_per_thread_17598),
                                                         np.int64(virt_num_groups_17599),
                                                         np.int64(threads_per_segment_17601),
                                                         group_res_arr_mem_17602,
                                                         mainzicounter_mem_17604)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_large_17575_var,
                                         ((np.long(num_groups_17177) * np.long(seghist_group_sizze_17175)),),
                                         (np.long(seghist_group_sizze_17175),))
              if synchronous:
                sync(self)
      max_num_groups_17636 = self.sizes["main.segscan_num_groups_17191"]
      num_groups_17192 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_16524,
                                                              sext_i32_i64(segscan_group_sizze_17190)),
                                                    sext_i32_i64(max_num_groups_17636))))
      mem_17354 = opencl_alloc(self, res_16524, "mem_17354")
      mem_17356 = opencl_alloc(self, bytes_17349, "mem_17356")
      if slt64(np.int64(0), res_16524):
        stage1_max_num_groups_17637 = self.max_group_size
        stage1_num_groups_17638 = smin64(stage1_max_num_groups_17637,
                                         num_groups_17192)
        num_threads_17639 = sext_i64_i32((stage1_num_groups_17638 * segscan_group_sizze_17190))
        if ((1 * (np.long(stage1_num_groups_17638) * np.long(segscan_group_sizze_17190))) != 0):
          self.mainziscan_stage1_17195_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_17190)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_17190)))),
                                                    np.int64(res_16524),
                                                    mem_17350, mem_17354,
                                                    mem_17356,
                                                    np.int32(num_threads_17639))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_17195_var,
                                     ((np.long(stage1_num_groups_17638) * np.long(segscan_group_sizze_17190)),),
                                     (np.long(segscan_group_sizze_17190),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_17638))) != 0):
          self.mainziscan_stage2_17195_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_17638)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_17638)))),
                                                    np.int64(res_16524),
                                                    mem_17354, mem_17356,
                                                    np.int64(stage1_num_groups_17638),
                                                    np.int32(num_threads_17639))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_17195_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_17638)),),
                                     (np.long(stage1_num_groups_17638),))
          if synchronous:
            sync(self)
        required_groups_17691 = sext_i64_i32(sdiv_up64(res_16524,
                                                       segscan_group_sizze_17190))
        if ((1 * (np.long(num_groups_17192) * np.long(segscan_group_sizze_17190))) != 0):
          self.mainziscan_stage3_17195_var.set_args(self.global_failure,
                                                    np.int64(res_16524),
                                                    np.int64(num_groups_17192),
                                                    mem_17354, mem_17356,
                                                    np.int32(num_threads_17639),
                                                    np.int32(required_groups_17691))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_17195_var,
                                     ((np.long(num_groups_17192) * np.long(segscan_group_sizze_17190)),),
                                     (np.long(segscan_group_sizze_17190),))
          if synchronous:
            sync(self)
      mem_17350 = None
      mem_17354 = None
      max_num_groups_17703 = self.sizes["main.segscan_num_groups_17199"]
      num_groups_17200 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_16524,
                                                              sext_i32_i64(segscan_group_sizze_17198)),
                                                    sext_i32_i64(max_num_groups_17703))))
      mem_17359 = opencl_alloc(self, res_16524, "mem_17359")
      mem_17361 = opencl_alloc(self, bytes_17349, "mem_17361")
      mem_17363 = opencl_alloc(self, res_16524, "mem_17363")
      if slt64(np.int64(0), res_16524):
        stage1_max_num_groups_17704 = self.max_group_size
        stage1_num_groups_17705 = smin64(stage1_max_num_groups_17704,
                                         num_groups_17200)
        num_threads_17706 = sext_i64_i32((stage1_num_groups_17705 * segscan_group_sizze_17198))
        if ((1 * (np.long(stage1_num_groups_17705) * np.long(segscan_group_sizze_17198))) != 0):
          self.mainziscan_stage1_17203_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_17198)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_17198)))),
                                                    np.int64(res_16524),
                                                    mem_17356, mem_17359,
                                                    mem_17361, mem_17363,
                                                    np.int32(num_threads_17706))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_17203_var,
                                     ((np.long(stage1_num_groups_17705) * np.long(segscan_group_sizze_17198)),),
                                     (np.long(segscan_group_sizze_17198),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_17705))) != 0):
          self.mainziscan_stage2_17203_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_17705)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_17705)))),
                                                    np.int64(res_16524),
                                                    mem_17359, mem_17361,
                                                    np.int64(stage1_num_groups_17705),
                                                    np.int32(num_threads_17706))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_17203_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_17705)),),
                                     (np.long(stage1_num_groups_17705),))
          if synchronous:
            sync(self)
        required_groups_17758 = sext_i64_i32(sdiv_up64(res_16524,
                                                       segscan_group_sizze_17198))
        if ((1 * (np.long(num_groups_17200) * np.long(segscan_group_sizze_17198))) != 0):
          self.mainziscan_stage3_17203_var.set_args(self.global_failure,
                                                    np.int64(res_16524),
                                                    np.int64(num_groups_17200),
                                                    mem_17359, mem_17361,
                                                    np.int32(num_threads_17706),
                                                    np.int32(required_groups_17758))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_17203_var,
                                     ((np.long(num_groups_17200) * np.long(segscan_group_sizze_17198)),),
                                     (np.long(segscan_group_sizze_17198),))
          if synchronous:
            sync(self)
      mem_17359 = None
      max_num_groups_17770 = self.sizes["main.segscan_num_groups_17207"]
      num_groups_17208 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_16524,
                                                              sext_i32_i64(segscan_group_sizze_17206)),
                                                    sext_i32_i64(max_num_groups_17770))))
      mem_17366 = opencl_alloc(self, res_16524, "mem_17366")
      bytes_17367 = (np.int64(4) * res_16524)
      mem_17368 = opencl_alloc(self, bytes_17367, "mem_17368")
      if slt64(np.int64(0), res_16524):
        stage1_max_num_groups_17771 = self.max_group_size
        stage1_num_groups_17772 = smin64(stage1_max_num_groups_17771,
                                         num_groups_17208)
        num_threads_17773 = sext_i64_i32((stage1_num_groups_17772 * segscan_group_sizze_17206))
        if ((1 * (np.long(stage1_num_groups_17772) * np.long(segscan_group_sizze_17206))) != 0):
          self.mainziscan_stage1_17211_var.set_args(self.global_failure,
                                                    self.failure_is_an_option,
                                                    self.global_failure_args,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * segscan_group_sizze_17206)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * segscan_group_sizze_17206)))),
                                                    np.int64(paths_16408),
                                                    np.float32(swap_term_16410),
                                                    np.int64(payments_16411),
                                                    np.float32(notional_16412),
                                                    np.float32(a_16413),
                                                    np.float32(b_16414),
                                                    np.float32(sigma_16415),
                                                    np.float32(res_16511),
                                                    np.int64(res_16524),
                                                    np.int64(i_17163),
                                                    mem_17336, mem_17356,
                                                    mem_17361, mem_17363,
                                                    mem_17366, mem_17368,
                                                    np.int32(num_threads_17773))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_17211_var,
                                     ((np.long(stage1_num_groups_17772) * np.long(segscan_group_sizze_17206)),),
                                     (np.long(segscan_group_sizze_17206),))
          if synchronous:
            sync(self)
        self.failure_is_an_option = np.int32(1)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_17772))) != 0):
          self.mainziscan_stage2_17211_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * stage1_num_groups_17772)))),
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(1) * stage1_num_groups_17772)))),
                                                    np.int64(res_16524),
                                                    mem_17366, mem_17368,
                                                    np.int64(stage1_num_groups_17772),
                                                    np.int32(num_threads_17773))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_17211_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_17772)),),
                                     (np.long(stage1_num_groups_17772),))
          if synchronous:
            sync(self)
        required_groups_17825 = sext_i64_i32(sdiv_up64(res_16524,
                                                       segscan_group_sizze_17206))
        if ((1 * (np.long(num_groups_17208) * np.long(segscan_group_sizze_17206))) != 0):
          self.mainziscan_stage3_17211_var.set_args(self.global_failure,
                                                    np.int64(res_16524),
                                                    np.int64(num_groups_17208),
                                                    mem_17366, mem_17368,
                                                    np.int32(num_threads_17773),
                                                    np.int32(required_groups_17825))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_17211_var,
                                     ((np.long(num_groups_17208) * np.long(segscan_group_sizze_17206)),),
                                     (np.long(segscan_group_sizze_17206),))
          if synchronous:
            sync(self)
      mem_17356 = None
      mem_17361 = None
      mem_17366 = None
      max_num_groups_17837 = self.sizes["main.segscan_num_groups_17259"]
      num_groups_17260 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(res_16524,
                                                              sext_i32_i64(segscan_group_sizze_17258)),
                                                    sext_i32_i64(max_num_groups_17837))))
      mem_17371 = opencl_alloc(self, bytes_17349, "mem_17371")
      if slt64(np.int64(0), res_16524):
        stage1_max_num_groups_17838 = self.max_group_size
        stage1_num_groups_17839 = smin64(stage1_max_num_groups_17838,
                                         num_groups_17260)
        num_threads_17840 = sext_i64_i32((stage1_num_groups_17839 * segscan_group_sizze_17258))
        if ((1 * (np.long(stage1_num_groups_17839) * np.long(segscan_group_sizze_17258))) != 0):
          self.mainziscan_stage1_17263_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * segscan_group_sizze_17258)))),
                                                    np.int64(res_16524),
                                                    mem_17363, mem_17371,
                                                    np.int32(num_threads_17840))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_17263_var,
                                     ((np.long(stage1_num_groups_17839) * np.long(segscan_group_sizze_17258)),),
                                     (np.long(segscan_group_sizze_17258),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int64(1)) * np.long(stage1_num_groups_17839))) != 0):
          self.mainziscan_stage2_17263_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(8) * stage1_num_groups_17839)))),
                                                    np.int64(res_16524),
                                                    mem_17371,
                                                    np.int64(stage1_num_groups_17839),
                                                    np.int32(num_threads_17840))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_17263_var,
                                     ((np.long(np.int64(1)) * np.long(stage1_num_groups_17839)),),
                                     (np.long(stage1_num_groups_17839),))
          if synchronous:
            sync(self)
        required_groups_17876 = sext_i64_i32(sdiv_up64(res_16524,
                                                       segscan_group_sizze_17258))
        if ((1 * (np.long(num_groups_17260) * np.long(segscan_group_sizze_17258))) != 0):
          self.mainziscan_stage3_17263_var.set_args(self.global_failure,
                                                    np.int64(res_16524),
                                                    np.int64(num_groups_17260),
                                                    mem_17371,
                                                    np.int32(num_threads_17840),
                                                    np.int32(required_groups_17876))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_17263_var,
                                     ((np.long(num_groups_17260) * np.long(segscan_group_sizze_17258)),),
                                     (np.long(segscan_group_sizze_17258),))
          if synchronous:
            sync(self)
      cond_16699 = slt64(np.int64(0), res_16524)
      if cond_16699:
        i_16701 = (res_16524 - np.int64(1))
        x_16702 = sle64(np.int64(0), i_16701)
        y_16703 = slt64(i_16701, res_16524)
        bounds_check_16704 = (x_16702 and y_16703)
        index_certs_16705 = True
        assert bounds_check_16704, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  cva.fut:135:36-88\n   #4  /prelude/soacs.fut:56:19-23\n   #5  /prelude/soacs.fut:56:3-37\n   #6  cva.fut:133:18-138:46\n   #7  cva.fut:105:1-145:18\n" % ("Index [",
                                                                                                                                                                                                                                                                                                                                                                                                          i_16701,
                                                                                                                                                                                                                                                                                                                                                                                                          "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                                                                                                                                                                          res_16524,
                                                                                                                                                                                                                                                                                                                                                                                                          "]."))
        read_res_17939 = np.empty(1, dtype=ct.c_int64)
        cl.enqueue_copy(self.queue, read_res_17939, mem_17371,
                        device_offset=(np.long(i_16701) * 8),
                        is_blocking=synchronous)
        sync(self)
        res_16706 = read_res_17939[0]
        num_segments_16700 = res_16706
      else:
        num_segments_16700 = np.int64(0)
      bounds_invalid_upwards_16707 = slt64(num_segments_16700, np.int64(0))
      valid_16708 = not(bounds_invalid_upwards_16707)
      range_valid_c_16709 = True
      assert valid_16708, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  cva.fut:135:36-88\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:133:18-138:46\n   #8  cva.fut:105:1-145:18\n" % ("Range ",
                                                                                                                                                                                                                                                                                                                                                                                                                                        np.int64(0),
                                                                                                                                                                                                                                                                                                                                                                                                                                        "..",
                                                                                                                                                                                                                                                                                                                                                                                                                                        np.int64(1),
                                                                                                                                                                                                                                                                                                                                                                                                                                        "..<",
                                                                                                                                                                                                                                                                                                                                                                                                                                        num_segments_16700,
                                                                                                                                                                                                                                                                                                                                                                                                                                        " is invalid."))
      bytes_17372 = (np.int64(4) * num_segments_16700)
      mem_17373 = opencl_alloc(self, bytes_17372, "mem_17373")
      self.futhark_builtinzhreplicate_f32(mem_17373, num_segments_16700,
                                          np.float32(0.0))
      segmap_group_sizze_17268 = self.sizes["main.segmap_group_size_17267"]
      segmap_usable_groups_17269 = sdiv_up64(res_16524,
                                             segmap_group_sizze_17268)
      if ((1 * (np.long(segmap_usable_groups_17269) * np.long(segmap_group_sizze_17268))) != 0):
        self.mainzisegmap_17265_var.set_args(self.global_failure,
                                             np.int64(res_16524),
                                             np.int64(num_segments_16700),
                                             mem_17363, mem_17368, mem_17371,
                                             mem_17373)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_17265_var,
                                   ((np.long(segmap_usable_groups_17269) * np.long(segmap_group_sizze_17268)),),
                                   (np.long(segmap_group_sizze_17268),))
        if synchronous:
          sync(self)
      mem_17363 = None
      mem_17368 = None
      mem_17371 = None
      max_num_groups_17902 = self.sizes["main.segred_num_groups_17273"]
      num_groups_17274 = sext_i64_i32(smax64(np.int64(1),
                                             smin64(sdiv_up64(num_segments_16700,
                                                              sext_i32_i64(segred_group_sizze_17272)),
                                                    sext_i32_i64(max_num_groups_17902))))
      mainzicounter_mem_17903 = self.mainzicounter_mem_17903
      group_res_arr_mem_17905 = opencl_alloc(self,
                                             (np.int32(4) * (segred_group_sizze_17272 * num_groups_17274)),
                                             "group_res_arr_mem_17905")
      num_threads_17907 = (num_groups_17274 * segred_group_sizze_17272)
      if ((1 * (np.long(num_groups_17274) * np.long(segred_group_sizze_17272))) != 0):
        self.mainzisegred_nonseg_17279_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long((np.int32(4) * segred_group_sizze_17272))),
                                                    cl.LocalMemory(np.long(np.int32(1))),
                                                    np.int64(num_segments_16700),
                                                    np.int64(num_groups_17274),
                                                    mem_17373, mem_17377,
                                                    mainzicounter_mem_17903,
                                                    group_res_arr_mem_17905,
                                                    np.int64(num_threads_17907))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegred_nonseg_17279_var,
                                   ((np.long(num_groups_17274) * np.long(segred_group_sizze_17272)),),
                                   (np.long(segred_group_sizze_17272),))
        if synchronous:
          sync(self)
      mem_17373 = None
      read_res_17941 = np.empty(1, dtype=ct.c_float)
      cl.enqueue_copy(self.queue, read_res_17941, mem_17377,
                      device_offset=(np.long(np.int64(0)) * 4),
                      is_blocking=synchronous)
      sync(self)
      res_16717 = read_res_17941[0]
      res_16723 = (res_16717 / res_16495)
      negate_arg_16724 = (a_16413 * res_16511)
      exp_arg_16725 = (np.float32(0.0) - negate_arg_16724)
      res_16726 = fpow32(np.float32(2.7182817459106445), exp_arg_16725)
      x_16727 = (np.float32(1.0) - res_16726)
      B_16728 = (x_16727 / a_16413)
      x_16729 = (B_16728 - res_16511)
      x_16730 = (y_16500 * x_16729)
      A1_16731 = (x_16730 / x_16496)
      y_16732 = fpow32(B_16728, np.float32(2.0))
      x_16733 = (x_16498 * y_16732)
      A2_16734 = (x_16733 / y_16501)
      exp_arg_16735 = (A1_16731 - A2_16734)
      res_16736 = fpow32(np.float32(2.7182817459106445), exp_arg_16735)
      negate_arg_16737 = (np.float32(5.000000074505806e-2) * B_16728)
      exp_arg_16738 = (np.float32(0.0) - negate_arg_16737)
      res_16739 = fpow32(np.float32(2.7182817459106445), exp_arg_16738)
      res_16740 = (res_16736 * res_16739)
      res_16741 = (res_16723 * res_16740)
      res_16507 = (res_16741 + redout_17161)
      cl.enqueue_copy(self.queue, mem_17338, np.array(res_16741,
                                                      dtype=ct.c_float),
                      device_offset=(np.long(i_17163) * 4),
                      is_blocking=synchronous)
      redout_tmp_17428 = res_16507
      redout_17161 = redout_tmp_17428
      i_17163 += one_17943
    res_16503 = redout_17161
    mem_17336 = None
    mem_17348 = None
    mem_17377 = None
    CVA_16742 = (np.float32(6.000000052154064e-3) * res_16503)
    mem_17384 = opencl_alloc(self, bytes_17320, "mem_17384")
    if ((steps_16409 * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_17384, mem_17338,
                      dest_offset=np.long(np.int64(0)),
                      src_offset=np.long(np.int64(0)),
                      byte_count=np.long((steps_16409 * np.int32(4))))
    if synchronous:
      sync(self)
    mem_17338 = None
    out_arrsizze_17403 = steps_16409
    out_mem_17402 = mem_17384
    scalar_out_17401 = CVA_16742
    return (scalar_out_17401, out_mem_17402, out_arrsizze_17403)
  def main(self, paths_16408_ext, steps_16409_ext, swap_term_16410_ext,
           payments_16411_ext, notional_16412_ext, a_16413_ext, b_16414_ext,
           sigma_16415_ext, r0_16416_ext):
    try:
      paths_16408 = np.int64(ct.c_int64(paths_16408_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(paths_16408_ext),
                                                                                                                            paths_16408_ext))
    try:
      steps_16409 = np.int64(ct.c_int64(steps_16409_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(steps_16409_ext),
                                                                                                                            steps_16409_ext))
    try:
      swap_term_16410 = np.float32(ct.c_float(swap_term_16410_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(swap_term_16410_ext),
                                                                                                                            swap_term_16410_ext))
    try:
      payments_16411 = np.int64(ct.c_int64(payments_16411_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(payments_16411_ext),
                                                                                                                            payments_16411_ext))
    try:
      notional_16412 = np.float32(ct.c_float(notional_16412_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(notional_16412_ext),
                                                                                                                            notional_16412_ext))
    try:
      a_16413 = np.float32(ct.c_float(a_16413_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(a_16413_ext),
                                                                                                                            a_16413_ext))
    try:
      b_16414 = np.float32(ct.c_float(b_16414_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(b_16414_ext),
                                                                                                                            b_16414_ext))
    try:
      sigma_16415 = np.float32(ct.c_float(sigma_16415_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(sigma_16415_ext),
                                                                                                                            sigma_16415_ext))
    try:
      r0_16416 = np.float32(ct.c_float(r0_16416_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(r0_16416_ext),
                                                                                                                            r0_16416_ext))
    (scalar_out_17401, out_mem_17402,
     out_arrsizze_17403) = self.futhark_main(paths_16408, steps_16409,
                                             swap_term_16410, payments_16411,
                                             notional_16412, a_16413, b_16414,
                                             sigma_16415, r0_16416)
    sync(self)
    return (np.float32(scalar_out_17401), cl.array.Array(self.queue,
                                                         (out_arrsizze_17403,),
                                                         ct.c_float,
                                                         data=out_mem_17402))