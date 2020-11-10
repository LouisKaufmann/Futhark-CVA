#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-label"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-label"
#endif
// Headers

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>


// Initialisation

struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);

// Arrays

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr);
const int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                                    struct futhark_f32_1d *arr);
struct futhark_f32_3d ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2);
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2);
int futhark_free_f32_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d *arr);
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data);
char *futhark_values_raw_f32_3d(struct futhark_context *ctx,
                                struct futhark_f32_3d *arr);
const int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                                    struct futhark_f32_3d *arr);
struct futhark_i64_1d ;
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const
                                          int64_t *data, int64_t dim0);
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0);
int futhark_free_i64_1d(struct futhark_context *ctx,
                        struct futhark_i64_1d *arr);
int futhark_values_i64_1d(struct futhark_context *ctx,
                          struct futhark_i64_1d *arr, int64_t *data);
char *futhark_values_raw_i64_1d(struct futhark_context *ctx,
                                struct futhark_i64_1d *arr);
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx,
                                    struct futhark_i64_1d *arr);

// Opaque values


// Entry points

int futhark_entry_main(struct futhark_context *ctx, float *out0,
                       struct futhark_f32_3d **out1, const int64_t in0, const
                       int64_t in1, const struct futhark_f32_1d *in2, const
                       struct futhark_i64_1d *in3, const
                       struct futhark_f32_1d *in4, const float in5, const
                       float in6, const float in7, const float in8);
int futhark_entry_test(struct futhark_context *ctx, float *out0, const
                       int64_t in0, const int64_t in1);
int futhark_entry_test2(struct futhark_context *ctx, float *out0, const
                        int64_t in0, const int64_t in1, const int64_t in2);

// Miscellaneous

int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
#define FUTHARK_BACKEND_c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
#include <stdarg.h>
// Start of util.h.
//
// Various helper functions that are useful in all generated C code.

#include <errno.h>
#include <string.h>

static const char *fut_progname = "(embedded Futhark)";

static void futhark_panic(int eval, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s: ", fut_progname);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(eval);
}

// For generating arbitrary-sized error messages.  It is the callers
// responsibility to free the buffer at some point.
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); // Must re-init.
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}


static inline void check_err(int errval, int sets_errno, const char *fun, int line,
                            const char *msg, ...)
{
  if (errval) {
    char str[256];
    char errnum[10];
    sprintf(errnum, "%d", errval);
    sprintf(str, "ERROR: %s in %s() at line %d with error code %s\n", msg, fun, line,
            sets_errno ? strerror(errno) : errnum);
    fprintf(stderr, "%s", str);
    exit(errval);
  }
}

#define CHECK_ERR(err, msg...) check_err(err, 0, __func__, __LINE__, msg)
#define CHECK_ERRNO(err, msg...) check_err(err, 1, __func__, __LINE__, msg)

// Read a file into a NUL-terminated string; returns NULL on error.
static void* slurp_file(const char *filename, size_t *size) {
  unsigned char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (unsigned char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

// Dump 'n' bytes from 'buf' into the file at the designated location.
// Returns 0 on success.
static int dump_file(const char *file, const void *buf, size_t n) {
  FILE *f = fopen(file, "w");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(buf, sizeof(char), n, f) != n) {
    return 1;
  }

  if (fclose(f) != 0) {
    return 1;
  }

  return 0;
}

struct str_builder {
  char *str;
  size_t capacity; // Size of buffer.
  size_t used; // Bytes used, *not* including final zero.
};

static void str_builder_init(struct str_builder *b) {
  b->capacity = 10;
  b->used = 0;
  b->str = malloc(b->capacity);
  b->str[0] = 0;
}

static void str_builder(struct str_builder *b, const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = (size_t)vsnprintf(NULL, 0, s, vl);

  while (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }

  va_start(vl, s); // Must re-init.
  vsnprintf(b->str+b->used, b->capacity-b->used, s, vl);
  b->used += needed;
}

// End of util.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
// Assuming POSIX

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

static int64_t get_wall_time_ns(void) {
  struct timespec time;
  assert(clock_gettime(CLOCK_REALTIME, &time) == 0);
  return time.tv_sec * 1000000000 + time.tv_nsec;
}


static inline uint64_t rdtsc() {
  unsigned int hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return  ((uint64_t) lo) | (((uint64_t) hi) << 32);
}

static inline void rdtsc_wait(uint64_t n) {
  const uint64_t start = rdtsc();
  while (rdtsc() < (start + n)) {
    __asm__("PAUSE");
  }
}
static inline void spin_for(uint64_t nb_cycles) {
  rdtsc_wait(nb_cycles);
}


#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, const void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    next_token(buf, bufsize);

    if (sscanf(buf, "%"SCNu64, (uint64_t*)&shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNi8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNu8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { futhark_panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      futhark_panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { futhark_panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  futhark_panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    futhark_panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    futhark_panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    futhark_panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    futhark_panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, stdin);
    if (ret != 1) {
      futhark_panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    futhark_panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, stdin);
  if (num_elems_read != elem_count) {
    futhark_panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int end_of_input() {
  skipspaces();
  char token[2];
  next_token(token, sizeof(token));
  if (strcmp(token, "") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 0; i < rank; i++) {
        printf("[%"PRIi64"]", shape[i]);
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      const unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type,
                       const void *data,
                       const int64_t *shape,
                       const int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    int64_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, (size_t)elem_size, 1, stdin);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

#define __private
static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"help", no_argument, NULL, 7}, {0,
                                                                            0,
                                                                            0,
                                                                            0}};
    static char *option_descriptions =
                "  -t/--write-runtime-to FILE Print the time taken to execute the program to the indicated file, an integral number of microseconds.\n  -r/--runs INT              Perform NUM runs of the program.\n  -D/--debugging             Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                   Print various low-overhead logging information to stderr while running.\n  -e/--entry-point NAME      The entry point to run. Defaults to main.\n  -b/--binary-output         Print the program result in the binary output format.\n  -h/--help                  Print help information and exit.\n";
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bh", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                futhark_panic(1, "Cannot open %s: %s\n", optarg,
                              strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                futhark_panic(1, "Need a positive number of runs, not %s\n",
                              optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'h') {
            printf("Usage: %s [OPTION]...\nOptions:\n\n%s\nFor more information, consult the Futhark User's Guide or the man pages.\n",
                   fut_progname, option_descriptions);
            exit(0);
        }
        if (ch == ':')
            futhark_panic(-1, "Missing argument for option %s\n", argv[optind -
                                                                       1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "  -t/--write-runtime-to FILE Print the time taken to execute the program to the indicated file, an integral number of microseconds.\n  -r/--runs INT              Perform NUM runs of the program.\n  -D/--debugging             Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                   Print various low-overhead logging information to stderr while running.\n  -e/--entry-point NAME      The entry point to run. Defaults to main.\n  -b/--binary-output         Print the program result in the binary output format.\n  -h/--help                  Print help information and exit.\n");
            futhark_panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    int64_t read_value_28913;
    
    if (read_scalar(&i64_info, &read_value_28913) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28914;
    
    if (read_scalar(&i64_info, &read_value_28914) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_28915;
    int64_t read_shape_28916[1];
    float *read_arr_28917 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_28917, read_shape_28916, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
                      "[]", f32_info.type_name, strerror(errno));
    
    struct futhark_i64_1d *read_value_28918;
    int64_t read_shape_28919[1];
    int64_t *read_arr_28920 = NULL;
    
    errno = 0;
    if (read_array(&i64_info, (void **) &read_arr_28920, read_shape_28919, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
                      "[]", i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_28921;
    int64_t read_shape_28922[1];
    float *read_arr_28923 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_28923, read_shape_28922, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4,
                      "[]", f32_info.type_name, strerror(errno));
    
    float read_value_28924;
    
    if (read_scalar(&f32_info, &read_value_28924) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      5, f32_info.type_name, strerror(errno));
    
    float read_value_28925;
    
    if (read_scalar(&f32_info, &read_value_28925) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      6, f32_info.type_name, strerror(errno));
    
    float read_value_28926;
    
    if (read_scalar(&f32_info, &read_value_28926) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      7, f32_info.type_name, strerror(errno));
    
    float read_value_28927;
    
    if (read_scalar(&f32_info, &read_value_28927) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      8, f32_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"main\"");
    
    float result_28928;
    struct futhark_f32_3d *result_28929;
    
    if (perform_warmup) {
        int r;
        
        ;
        ;
        assert((read_value_28915 = futhark_new_f32_1d(ctx, read_arr_28917,
                                                      read_shape_28916[0])) !=
            0);
        assert((read_value_28918 = futhark_new_i64_1d(ctx, read_arr_28920,
                                                      read_shape_28919[0])) !=
            0);
        assert((read_value_28921 = futhark_new_f32_1d(ctx, read_arr_28923,
                                                      read_shape_28922[0])) !=
            0);
        ;
        ;
        ;
        ;
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_28928, &result_28929,
                               read_value_28913, read_value_28914,
                               read_value_28915, read_value_28918,
                               read_value_28921, read_value_28924,
                               read_value_28925, read_value_28926,
                               read_value_28927);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        ;
        ;
        assert(futhark_free_f32_1d(ctx, read_value_28915) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_28918) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_28921) == 0);
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_f32_3d(ctx, result_28929) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        ;
        assert((read_value_28915 = futhark_new_f32_1d(ctx, read_arr_28917,
                                                      read_shape_28916[0])) !=
            0);
        assert((read_value_28918 = futhark_new_i64_1d(ctx, read_arr_28920,
                                                      read_shape_28919[0])) !=
            0);
        assert((read_value_28921 = futhark_new_f32_1d(ctx, read_arr_28923,
                                                      read_shape_28922[0])) !=
            0);
        ;
        ;
        ;
        ;
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_28928, &result_28929,
                               read_value_28913, read_value_28914,
                               read_value_28915, read_value_28918,
                               read_value_28921, read_value_28924,
                               read_value_28925, read_value_28926,
                               read_value_28927);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        ;
        ;
        assert(futhark_free_f32_1d(ctx, read_value_28915) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_28918) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_28921) == 0);
        ;
        ;
        ;
        ;
        if (run < num_runs - 1) {
            ;
            assert(futhark_free_f32_3d(ctx, result_28929) == 0);
        }
    }
    ;
    ;
    free(read_arr_28917);
    free(read_arr_28920);
    free(read_arr_28923);
    ;
    ;
    ;
    ;
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &f32_info, &result_28928);
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_28929)[0] *
                            futhark_shape_f32_3d(ctx, result_28929)[1] *
                            futhark_shape_f32_3d(ctx, result_28929)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_28929, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_28929), 3);
        free(arr);
    }
    printf("\n");
    ;
    assert(futhark_free_f32_3d(ctx, result_28929) == 0);
}
static void futrts_cli_entry_test(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    int64_t read_value_28930;
    
    if (read_scalar(&i64_info, &read_value_28930) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28931;
    
    if (read_scalar(&i64_info, &read_value_28931) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"test\"");
    
    float result_28932;
    
    if (perform_warmup) {
        int r;
        
        ;
        ;
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_test(ctx, &result_28932, read_value_28930,
                               read_value_28931);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        ;
        ;
        ;
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        ;
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_test(ctx, &result_28932, read_value_28930,
                               read_value_28931);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        ;
        ;
        if (run < num_runs - 1) {
            ;
        }
    }
    ;
    ;
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &f32_info, &result_28932);
    printf("\n");
    ;
}
static void futrts_cli_entry_test2(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    int64_t read_value_28933;
    
    if (read_scalar(&i64_info, &read_value_28933) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28934;
    
    if (read_scalar(&i64_info, &read_value_28934) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28935;
    
    if (read_scalar(&i64_info, &read_value_28935) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      2, i64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"test2\"");
    
    float result_28936;
    
    if (perform_warmup) {
        int r;
        
        ;
        ;
        ;
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_test2(ctx, &result_28936, read_value_28933,
                                read_value_28934, read_value_28935);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        ;
        ;
        ;
        ;
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        ;
        ;
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_test2(ctx, &result_28936, read_value_28933,
                                read_value_28934, read_value_28935);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        ;
        ;
        ;
        if (run < num_runs - 1) {
            ;
        }
    }
    ;
    ;
    ;
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &f32_info, &result_28936);
    printf("\n");
    ;
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}, {.name =
                                                                         "test",
                                                                         .fun =
                                                                         futrts_cli_entry_test},
                                               {.name ="test2", .fun =
                                                futrts_cli_entry_test2}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        futhark_panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    
    char *error = futhark_context_get_error(ctx);
    
    if (error != NULL)
        futhark_panic(1, "%s", error);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        
        char *report = futhark_context_report(ctx);
        
        fputs(report, stderr);
        free(report);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>

// Start of lock.h.

// A very simple cross-platform implementation of locks.  Uses
// pthreads on Unix and some Windows thing there.  Futhark's
// host-level code is not multithreaded, but user code may be, so we
// need some mechanism for ensuring atomic access to API functions.
// This is that mechanism.  It is not exposed to user code at all, so
// we do not have to worry about name collisions.

#ifdef _WIN32

typedef HANDLE lock_t;

static void create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  // Default security attributes.
                      FALSE, // Initially unlocked.
                      NULL); // Unnamed.
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
// Assuming POSIX

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  // Nothing to do for pthreads.
  (void)lock;
}

#endif

// End of lock.h.

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
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
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
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
int init_constants(struct futhark_context *);
int free_constants(struct futhark_context *);
static float testzistatic_array_realtype_28889[45] = {1.0F, -0.5F, 1.0F, 1.0F,
                                                      1.0F, 1.0F, 1.0F, 1.0F,
                                                      1.0F, 1.0F, -0.5F, 1.0F,
                                                      1.0F, 1.0F, 1.0F, 1.0F,
                                                      1.0F, 1.0F, 1.0F, -0.5F,
                                                      1.0F, 1.0F, 1.0F, 1.0F,
                                                      1.0F, 1.0F, 1.0F, 1.0F,
                                                      -0.5F, 1.0F, 1.0F, 1.0F,
                                                      1.0F, 1.0F, 1.0F, 1.0F,
                                                      1.0F, -0.5F, 1.0F, 1.0F,
                                                      1.0F, 1.0F, 1.0F, 1.0F,
                                                      1.0F};
static int64_t testzistatic_array_realtype_28890[45] = {10, 20, 5, 5, 50, 20,
                                                        30, 15, 18, 10, 200, 5,
                                                        5, 50, 20, 30, 15, 18,
                                                        10, 20, 5, 5, 100, 20,
                                                        30, 15, 18, 10, 20, 5,
                                                        5, 50, 20, 30, 15, 18,
                                                        10, 20, 5, 5, 50, 20,
                                                        30, 15, 18};
static float testzistatic_array_realtype_28891[45] = {1.0F, 0.5F, 0.25F, 0.1F,
                                                      0.3F, 0.1F, 2.0F, 3.0F,
                                                      1.0F, 1.0F, 0.5F, 0.25F,
                                                      0.1F, 0.3F, 0.1F, 2.0F,
                                                      3.0F, 1.0F, 1.0F, 0.5F,
                                                      0.25F, 0.1F, 0.3F, 0.1F,
                                                      2.0F, 3.0F, 1.0F, 1.0F,
                                                      0.5F, 0.25F, 0.1F, 0.3F,
                                                      0.1F, 2.0F, 3.0F, 1.0F,
                                                      1.0F, 0.5F, 0.25F, 0.1F,
                                                      0.3F, 0.1F, 2.0F, 3.0F,
                                                      1.0F};
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
struct futhark_context_config {
    int debugging;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->debugging = 0;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int detail)
{
    cfg->debugging = detail;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int detail)
{
    /* Does nothing for this backend. */
    (void) cfg;
    (void) detail;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    lock_t lock;
    char *error;
    int profiling_paused;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct {
        int dummy;
    } constants;
    struct memblock testzistatic_array_28809;
    struct memblock testzistatic_array_28810;
    struct memblock testzistatic_array_28811;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->detail_memory = cfg->debugging;
    ctx->debugging = cfg->debugging;
    ctx->profiling = cfg->debugging;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->testzistatic_array_28809 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_28889,
                                                       0};
    ctx->testzistatic_array_28810 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_28890,
                                                       0};
    ctx->testzistatic_array_28811 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_28891,
                                                       0};
    init_constants(ctx);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_constants(ctx);
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "default space",
                      ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
char *futhark_context_report(struct futhark_context *ctx)
{
    struct str_builder builder;
    
    str_builder_init(&builder);
    if (ctx->detail_memory || ctx->profiling) {
        { }
    }
    if (ctx->profiling) { }
    return builder.str;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 1;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 0;
}
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_28845,
                       struct memblock *out_mem_p_28846,
                       int64_t *out_out_arrsizze_28847,
                       int64_t *out_out_arrsizze_28848,
                       int64_t *out_out_arrsizze_28849,
                       struct memblock swap_term_mem_28510,
                       struct memblock payments_mem_28511,
                       struct memblock notional_mem_28512, int64_t n_26622,
                       int64_t n_26623, int64_t n_26624, int64_t paths_26625,
                       int64_t steps_26626, float a_26630, float b_26631,
                       float sigma_26632, float r0_26633);
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_28868,
                       int64_t paths_27317, int64_t steps_27318);
static int futrts_test2(struct futhark_context *ctx,
                        float *out_scalar_out_28892, int64_t paths_27744,
                        int64_t steps_27745, int64_t numswaps_27746);
int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    
  cleanup:
    return err;
}
int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_28845,
                       struct memblock *out_mem_p_28846,
                       int64_t *out_out_arrsizze_28847,
                       int64_t *out_out_arrsizze_28848,
                       int64_t *out_out_arrsizze_28849,
                       struct memblock swap_term_mem_28510,
                       struct memblock payments_mem_28511,
                       struct memblock notional_mem_28512, int64_t n_26622,
                       int64_t n_26623, int64_t n_26624, int64_t paths_26625,
                       int64_t steps_26626, float a_26630, float b_26631,
                       float sigma_26632, float r0_26633)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_28514_cached_sizze_28850 = 0;
    char *mem_28514 = NULL;
    size_t mem_28516_cached_sizze_28851 = 0;
    char *mem_28516 = NULL;
    size_t mem_28518_cached_sizze_28852 = 0;
    char *mem_28518 = NULL;
    size_t mem_28520_cached_sizze_28853 = 0;
    char *mem_28520 = NULL;
    size_t mem_28571_cached_sizze_28854 = 0;
    char *mem_28571 = NULL;
    size_t mem_28583_cached_sizze_28855 = 0;
    char *mem_28583 = NULL;
    size_t mem_28597_cached_sizze_28856 = 0;
    char *mem_28597 = NULL;
    size_t mem_28627_cached_sizze_28857 = 0;
    char *mem_28627 = NULL;
    size_t mem_28644_cached_sizze_28858 = 0;
    char *mem_28644 = NULL;
    size_t mem_28656_cached_sizze_28859 = 0;
    char *mem_28656 = NULL;
    size_t mem_28666_cached_sizze_28860 = 0;
    char *mem_28666 = NULL;
    size_t mem_28668_cached_sizze_28861 = 0;
    char *mem_28668 = NULL;
    size_t mem_28694_cached_sizze_28862 = 0;
    char *mem_28694 = NULL;
    size_t mem_28708_cached_sizze_28863 = 0;
    char *mem_28708 = NULL;
    size_t mem_28722_cached_sizze_28864 = 0;
    char *mem_28722 = NULL;
    size_t mem_28724_cached_sizze_28865 = 0;
    char *mem_28724 = NULL;
    size_t mem_28750_cached_sizze_28866 = 0;
    char *mem_28750 = NULL;
    size_t mem_28764_cached_sizze_28867 = 0;
    char *mem_28764 = NULL;
    float scalar_out_28808;
    struct memblock out_mem_28809;
    
    out_mem_28809.references = NULL;
    
    int64_t out_arrsizze_28810;
    int64_t out_arrsizze_28811;
    int64_t out_arrsizze_28812;
    bool dim_match_26634 = n_26622 == n_26623;
    bool empty_or_match_cert_26635;
    
    if (!dim_match_26634) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cva.fut:102:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_26636 = n_26622 == n_26624;
    bool empty_or_match_cert_26637;
    
    if (!dim_match_26636) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cva.fut:102:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
            return 1;
        return 1;
    }
    
    float res_26639;
    float redout_28227 = -INFINITY;
    
    for (int64_t i_28228 = 0; i_28228 < n_26622; i_28228++) {
        float x_26643 = ((float *) swap_term_mem_28510.mem)[i_28228];
        int64_t x_26644 = ((int64_t *) payments_mem_28511.mem)[i_28228];
        float res_26645 = sitofp_i64_f32(x_26644);
        float res_26646 = x_26643 * res_26645;
        float res_26642 = fmax32(res_26646, redout_28227);
        float redout_tmp_28813 = res_26642;
        
        redout_28227 = redout_tmp_28813;
    }
    res_26639 = redout_28227;
    
    float res_26647 = sitofp_i64_f32(steps_26626);
    float dt_26648 = res_26639 / res_26647;
    float x_26650 = fpow32(a_26630, 2.0F);
    float x_26651 = b_26631 * x_26650;
    float x_26652 = fpow32(sigma_26632, 2.0F);
    float y_26653 = x_26652 / 2.0F;
    float y_26654 = x_26651 - y_26653;
    float y_26655 = 4.0F * a_26630;
    int64_t bytes_28513 = 4 * n_26622;
    
    if (mem_28514_cached_sizze_28850 < (size_t) bytes_28513) {
        mem_28514 = realloc(mem_28514, bytes_28513);
        mem_28514_cached_sizze_28850 = bytes_28513;
    }
    if (mem_28516_cached_sizze_28851 < (size_t) bytes_28513) {
        mem_28516 = realloc(mem_28516, bytes_28513);
        mem_28516_cached_sizze_28851 = bytes_28513;
    }
    
    int64_t bytes_28517 = 8 * n_26622;
    
    if (mem_28518_cached_sizze_28852 < (size_t) bytes_28517) {
        mem_28518 = realloc(mem_28518, bytes_28517);
        mem_28518_cached_sizze_28852 = bytes_28517;
    }
    if (mem_28520_cached_sizze_28853 < (size_t) bytes_28513) {
        mem_28520 = realloc(mem_28520, bytes_28513);
        mem_28520_cached_sizze_28853 = bytes_28513;
    }
    for (int64_t i_28239 = 0; i_28239 < n_26622; i_28239++) {
        float res_26665 = ((float *) swap_term_mem_28510.mem)[i_28239];
        int64_t res_26666 = ((int64_t *) payments_mem_28511.mem)[i_28239];
        bool bounds_invalid_upwards_26668 = slt64(res_26666, 1);
        bool valid_26669 = !bounds_invalid_upwards_26668;
        bool range_valid_c_26670;
        
        if (!valid_26669) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_26666,
                          " is invalid.",
                          "-> #0  cva.fut:55:29-48\n   #1  cva.fut:96:25-65\n   #2  cva.fut:111:16-62\n   #3  cva.fut:107:17-111:85\n   #4  cva.fut:102:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                return 1;
            return 1;
        }
        
        bool y_26672 = slt64(0, res_26666);
        bool index_certs_26673;
        
        if (!y_26672) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_26666, "].",
                                   "-> #0  cva.fut:97:47-70\n   #1  cva.fut:111:16-62\n   #2  cva.fut:107:17-111:85\n   #3  cva.fut:102:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                return 1;
            return 1;
        }
        
        float binop_y_26674 = sitofp_i64_f32(res_26666);
        float index_primexp_26675 = res_26665 * binop_y_26674;
        float negate_arg_26676 = a_26630 * index_primexp_26675;
        float exp_arg_26677 = 0.0F - negate_arg_26676;
        float res_26678 = fpow32(2.7182817F, exp_arg_26677);
        float x_26679 = 1.0F - res_26678;
        float B_26680 = x_26679 / a_26630;
        float x_26681 = B_26680 - index_primexp_26675;
        float x_26682 = y_26654 * x_26681;
        float A1_26683 = x_26682 / x_26650;
        float y_26684 = fpow32(B_26680, 2.0F);
        float x_26685 = x_26652 * y_26684;
        float A2_26686 = x_26685 / y_26655;
        float exp_arg_26687 = A1_26683 - A2_26686;
        float res_26688 = fpow32(2.7182817F, exp_arg_26687);
        float negate_arg_26689 = r0_26633 * B_26680;
        float exp_arg_26690 = 0.0F - negate_arg_26689;
        float res_26691 = fpow32(2.7182817F, exp_arg_26690);
        float res_26692 = res_26688 * res_26691;
        float res_26693;
        float redout_28229 = 0.0F;
        
        for (int64_t i_28230 = 0; i_28230 < res_26666; i_28230++) {
            int64_t index_primexp_28364 = add64(1, i_28230);
            float res_26698 = sitofp_i64_f32(index_primexp_28364);
            float res_26699 = res_26665 * res_26698;
            float negate_arg_26700 = a_26630 * res_26699;
            float exp_arg_26701 = 0.0F - negate_arg_26700;
            float res_26702 = fpow32(2.7182817F, exp_arg_26701);
            float x_26703 = 1.0F - res_26702;
            float B_26704 = x_26703 / a_26630;
            float x_26705 = B_26704 - res_26699;
            float x_26706 = y_26654 * x_26705;
            float A1_26707 = x_26706 / x_26650;
            float y_26708 = fpow32(B_26704, 2.0F);
            float x_26709 = x_26652 * y_26708;
            float A2_26710 = x_26709 / y_26655;
            float exp_arg_26711 = A1_26707 - A2_26710;
            float res_26712 = fpow32(2.7182817F, exp_arg_26711);
            float negate_arg_26713 = r0_26633 * B_26704;
            float exp_arg_26714 = 0.0F - negate_arg_26713;
            float res_26715 = fpow32(2.7182817F, exp_arg_26714);
            float res_26716 = res_26712 * res_26715;
            float res_26696 = res_26716 + redout_28229;
            float redout_tmp_28818 = res_26696;
            
            redout_28229 = redout_tmp_28818;
        }
        res_26693 = redout_28229;
        
        float x_26717 = 1.0F - res_26692;
        float y_26718 = res_26665 * res_26693;
        float res_26719 = x_26717 / y_26718;
        
        ((float *) mem_28514)[i_28239] = res_26719;
        memmove(mem_28516 + i_28239 * 4, notional_mem_28512.mem + i_28239 * 4,
                (int32_t) sizeof(float));
        memmove(mem_28518 + i_28239 * 8, payments_mem_28511.mem + i_28239 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_28520 + i_28239 * 4, swap_term_mem_28510.mem + i_28239 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_26720 = res_26647 / res_26639;
    bool bounds_invalid_upwards_26721 = slt64(steps_26626, 1);
    bool valid_26722 = !bounds_invalid_upwards_26721;
    bool range_valid_c_26723;
    
    if (!valid_26722) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_26626,
                               " is invalid.",
                               "-> #0  cva.fut:61:56-67\n   #1  cva.fut:113:17-44\n   #2  cva.fut:102:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
            return 1;
        return 1;
    }
    
    bool bounds_invalid_upwards_26725 = slt64(paths_26625, 0);
    bool valid_26726 = !bounds_invalid_upwards_26725;
    bool range_valid_c_26727;
    
    if (!valid_26726) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_26625,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:116:19-49\n   #4  cva.fut:102:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
            return 1;
        return 1;
    }
    
    int64_t upper_bound_26730 = sub64(steps_26626, 1);
    float res_26731;
    
    res_26731 = futrts_sqrt32(dt_26648);
    
    int64_t binop_x_28570 = paths_26625 * steps_26626;
    int64_t bytes_28569 = 4 * binop_x_28570;
    
    if (mem_28571_cached_sizze_28854 < (size_t) bytes_28569) {
        mem_28571 = realloc(mem_28571, bytes_28569);
        mem_28571_cached_sizze_28854 = bytes_28569;
    }
    
    int64_t bytes_28582 = 4 * steps_26626;
    
    if (mem_28583_cached_sizze_28855 < (size_t) bytes_28582) {
        mem_28583 = realloc(mem_28583, bytes_28582);
        mem_28583_cached_sizze_28855 = bytes_28582;
    }
    if (mem_28597_cached_sizze_28856 < (size_t) bytes_28582) {
        mem_28597 = realloc(mem_28597, bytes_28582);
        mem_28597_cached_sizze_28856 = bytes_28582;
    }
    for (int64_t i_28819 = 0; i_28819 < steps_26626; i_28819++) {
        ((float *) mem_28597)[i_28819] = r0_26633;
    }
    for (int64_t i_28250 = 0; i_28250 < paths_26625; i_28250++) {
        int32_t res_26734 = sext_i64_i32(i_28250);
        int32_t x_26735 = lshr32(res_26734, 16);
        int32_t x_26736 = res_26734 ^ x_26735;
        int32_t x_26737 = mul32(73244475, x_26736);
        int32_t x_26738 = lshr32(x_26737, 16);
        int32_t x_26739 = x_26737 ^ x_26738;
        int32_t x_26740 = mul32(73244475, x_26739);
        int32_t x_26741 = lshr32(x_26740, 16);
        int32_t x_26742 = x_26740 ^ x_26741;
        int32_t unsign_arg_26743 = 777822902 ^ x_26742;
        int32_t unsign_arg_26744 = mul32(48271, unsign_arg_26743);
        int32_t unsign_arg_26745 = umod32(unsign_arg_26744, 2147483647);
        
        for (int64_t i_28246 = 0; i_28246 < steps_26626; i_28246++) {
            int32_t res_26748 = sext_i64_i32(i_28246);
            int32_t x_26749 = lshr32(res_26748, 16);
            int32_t x_26750 = res_26748 ^ x_26749;
            int32_t x_26751 = mul32(73244475, x_26750);
            int32_t x_26752 = lshr32(x_26751, 16);
            int32_t x_26753 = x_26751 ^ x_26752;
            int32_t x_26754 = mul32(73244475, x_26753);
            int32_t x_26755 = lshr32(x_26754, 16);
            int32_t x_26756 = x_26754 ^ x_26755;
            int32_t unsign_arg_26757 = unsign_arg_26745 ^ x_26756;
            int32_t unsign_arg_26758 = mul32(48271, unsign_arg_26757);
            int32_t unsign_arg_26759 = umod32(unsign_arg_26758, 2147483647);
            int32_t unsign_arg_26760 = mul32(48271, unsign_arg_26759);
            int32_t unsign_arg_26761 = umod32(unsign_arg_26760, 2147483647);
            float res_26762 = uitofp_i32_f32(unsign_arg_26759);
            float res_26763 = res_26762 / 2.1474836e9F;
            float res_26764 = uitofp_i32_f32(unsign_arg_26761);
            float res_26765 = res_26764 / 2.1474836e9F;
            float res_26766;
            
            res_26766 = futrts_log32(res_26763);
            
            float res_26767 = -2.0F * res_26766;
            float res_26768;
            
            res_26768 = futrts_sqrt32(res_26767);
            
            float res_26769 = 6.2831855F * res_26765;
            float res_26770;
            
            res_26770 = futrts_cos32(res_26769);
            
            float res_26771 = res_26768 * res_26770;
            
            ((float *) mem_28583)[i_28246] = res_26771;
        }
        memmove(mem_28571 + i_28250 * steps_26626 * 4, mem_28597 + 0,
                steps_26626 * (int32_t) sizeof(float));
        for (int64_t i_26774 = 0; i_26774 < upper_bound_26730; i_26774++) {
            bool y_26776 = slt64(i_26774, steps_26626);
            bool index_certs_26777;
            
            if (!y_26776) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_26774,
                              "] out of bounds for array of shape [",
                              steps_26626, "].",
                              "-> #0  cva.fut:72:97-104\n   #1  cva.fut:124:32-62\n   #2  cva.fut:124:22-69\n   #3  cva.fut:102:1-147:20\n");
                if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                    return 1;
                return 1;
            }
            
            float shortstep_arg_26778 = ((float *) mem_28583)[i_26774];
            float shortstep_arg_26779 = ((float *) mem_28571)[i_28250 *
                                                              steps_26626 +
                                                              i_26774];
            float y_26780 = b_26631 - shortstep_arg_26779;
            float x_26781 = a_26630 * y_26780;
            float x_26782 = dt_26648 * x_26781;
            float x_26783 = res_26731 * shortstep_arg_26778;
            float y_26784 = sigma_26632 * x_26783;
            float delta_r_26785 = x_26782 + y_26784;
            float res_26786 = shortstep_arg_26779 + delta_r_26785;
            int64_t i_26787 = add64(1, i_26774);
            bool x_26788 = sle64(0, i_26787);
            bool y_26789 = slt64(i_26787, steps_26626);
            bool bounds_check_26790 = x_26788 && y_26789;
            bool index_certs_26791;
            
            if (!bounds_check_26790) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_26787,
                              "] out of bounds for array of shape [",
                              steps_26626, "].",
                              "-> #0  cva.fut:72:58-105\n   #1  cva.fut:124:32-62\n   #2  cva.fut:124:22-69\n   #3  cva.fut:102:1-147:20\n");
                if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                    return 1;
                return 1;
            }
            ((float *) mem_28571)[i_28250 * steps_26626 + i_26787] = res_26786;
        }
    }
    
    int64_t flat_dim_26795 = n_26622 * paths_26625;
    float res_27039 = sitofp_i64_f32(paths_26625);
    int64_t binop_x_28625 = paths_26625 * steps_26626;
    int64_t binop_x_28626 = n_26622 * binop_x_28625;
    int64_t bytes_28624 = 4 * binop_x_28626;
    
    if (mem_28627_cached_sizze_28857 < (size_t) bytes_28624) {
        mem_28627 = realloc(mem_28627, bytes_28624);
        mem_28627_cached_sizze_28857 = bytes_28624;
    }
    
    int64_t binop_x_28643 = n_26622 * paths_26625;
    int64_t bytes_28642 = 4 * binop_x_28643;
    
    if (mem_28644_cached_sizze_28858 < (size_t) bytes_28642) {
        mem_28644 = realloc(mem_28644, bytes_28642);
        mem_28644_cached_sizze_28858 = bytes_28642;
    }
    if (mem_28656_cached_sizze_28859 < (size_t) bytes_28513) {
        mem_28656 = realloc(mem_28656, bytes_28513);
        mem_28656_cached_sizze_28859 = bytes_28513;
    }
    
    int64_t bytes_28665 = 8 * flat_dim_26795;
    
    if (mem_28666_cached_sizze_28860 < (size_t) bytes_28665) {
        mem_28666 = realloc(mem_28666, bytes_28665);
        mem_28666_cached_sizze_28860 = bytes_28665;
    }
    if (mem_28668_cached_sizze_28861 < (size_t) bytes_28665) {
        mem_28668 = realloc(mem_28668, bytes_28665);
        mem_28668_cached_sizze_28861 = bytes_28665;
    }
    
    float res_27041;
    float redout_28353 = 0.0F;
    
    for (int64_t i_28355 = 0; i_28355 < steps_26626; i_28355++) {
        int64_t index_primexp_28501 = add64(1, i_28355);
        float res_27048 = sitofp_i64_f32(index_primexp_28501);
        float res_27049 = res_27048 / sims_per_year_26720;
        
        for (int64_t i_28254 = 0; i_28254 < paths_26625; i_28254++) {
            float x_27051 = ((float *) mem_28571)[i_28254 * steps_26626 +
                                                  i_28355];
            
            for (int64_t i_28826 = 0; i_28826 < n_26622; i_28826++) {
                ((float *) mem_28656)[i_28826] = x_27051;
            }
            memmove(mem_28644 + i_28254 * n_26622 * 4, mem_28656 + 0, n_26622 *
                    (int32_t) sizeof(float));
        }
        
        int64_t discard_28264;
        int64_t scanacc_28258 = 0;
        
        for (int64_t i_28261 = 0; i_28261 < flat_dim_26795; i_28261++) {
            int64_t binop_x_28375 = squot64(i_28261, n_26622);
            int64_t binop_y_28376 = n_26622 * binop_x_28375;
            int64_t new_index_28377 = i_28261 - binop_y_28376;
            int64_t x_27058 = ((int64_t *) mem_28518)[new_index_28377];
            float x_27059 = ((float *) mem_28520)[new_index_28377];
            float x_27060 = res_27049 / x_27059;
            float ceil_arg_27061 = x_27060 - 1.0F;
            float res_27062;
            
            res_27062 = futrts_ceil32(ceil_arg_27061);
            
            int64_t res_27063 = fptosi_f32_i64(res_27062);
            int64_t max_arg_27064 = sub64(x_27058, res_27063);
            int64_t res_27065 = smax64(0, max_arg_27064);
            bool cond_27066 = res_27065 == 0;
            int64_t res_27067;
            
            if (cond_27066) {
                res_27067 = 1;
            } else {
                res_27067 = res_27065;
            }
            
            int64_t res_27057 = add64(res_27067, scanacc_28258);
            
            ((int64_t *) mem_28666)[i_28261] = res_27057;
            ((int64_t *) mem_28668)[i_28261] = res_27067;
            
            int64_t scanacc_tmp_28827 = res_27057;
            
            scanacc_28258 = scanacc_tmp_28827;
        }
        discard_28264 = scanacc_28258;
        
        int64_t res_27069;
        int64_t redout_28265 = 0;
        
        for (int64_t i_28266 = 0; i_28266 < flat_dim_26795; i_28266++) {
            int64_t x_27073 = ((int64_t *) mem_28668)[i_28266];
            int64_t res_27072 = add64(x_27073, redout_28265);
            int64_t redout_tmp_28830 = res_27072;
            
            redout_28265 = redout_tmp_28830;
        }
        res_27069 = redout_28265;
        
        bool bounds_invalid_upwards_27074 = slt64(res_27069, 0);
        bool valid_27075 = !bounds_invalid_upwards_27074;
        bool range_valid_c_27076;
        
        if (!valid_27075) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", res_27069,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:102:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                return 1;
            return 1;
        }
        
        int64_t bytes_28693 = 8 * res_27069;
        
        if (mem_28694_cached_sizze_28862 < (size_t) bytes_28693) {
            mem_28694 = realloc(mem_28694, bytes_28693);
            mem_28694_cached_sizze_28862 = bytes_28693;
        }
        for (int64_t i_28831 = 0; i_28831 < res_27069; i_28831++) {
            ((int64_t *) mem_28694)[i_28831] = 0;
        }
        for (int64_t iter_28267 = 0; iter_28267 < flat_dim_26795;
             iter_28267++) {
            int64_t i_p_o_28382 = add64(-1, iter_28267);
            int64_t rot_i_28383 = smod64(i_p_o_28382, flat_dim_26795);
            int64_t pixel_28270 = ((int64_t *) mem_28666)[rot_i_28383];
            bool cond_27084 = iter_28267 == 0;
            int64_t res_27085;
            
            if (cond_27084) {
                res_27085 = 0;
            } else {
                res_27085 = pixel_28270;
            }
            
            bool less_than_zzero_28271 = slt64(res_27085, 0);
            bool greater_than_sizze_28272 = sle64(res_27069, res_27085);
            bool outside_bounds_dim_28273 = less_than_zzero_28271 ||
                 greater_than_sizze_28272;
            
            if (!outside_bounds_dim_28273) {
                int64_t read_hist_28275 = ((int64_t *) mem_28694)[res_27085];
                int64_t res_27081 = smax64(iter_28267, read_hist_28275);
                
                ((int64_t *) mem_28694)[res_27085] = res_27081;
            }
        }
        if (mem_28708_cached_sizze_28863 < (size_t) bytes_28693) {
            mem_28708 = realloc(mem_28708, bytes_28693);
            mem_28708_cached_sizze_28863 = bytes_28693;
        }
        
        int64_t discard_28288;
        int64_t scanacc_28281 = 0;
        
        for (int64_t i_28284 = 0; i_28284 < res_27069; i_28284++) {
            int64_t x_27095 = ((int64_t *) mem_28694)[i_28284];
            bool res_27096 = slt64(0, x_27095);
            int64_t res_27093;
            
            if (res_27096) {
                res_27093 = x_27095;
            } else {
                int64_t res_27094 = add64(x_27095, scanacc_28281);
                
                res_27093 = res_27094;
            }
            ((int64_t *) mem_28708)[i_28284] = res_27093;
            
            int64_t scanacc_tmp_28833 = res_27093;
            
            scanacc_28281 = scanacc_tmp_28833;
        }
        discard_28288 = scanacc_28281;
        
        int64_t bytes_28721 = 4 * res_27069;
        
        if (mem_28722_cached_sizze_28864 < (size_t) bytes_28721) {
            mem_28722 = realloc(mem_28722, bytes_28721);
            mem_28722_cached_sizze_28864 = bytes_28721;
        }
        if (mem_28724_cached_sizze_28865 < (size_t) res_27069) {
            mem_28724 = realloc(mem_28724, res_27069);
            mem_28724_cached_sizze_28865 = res_27069;
        }
        
        int64_t inpacc_27105;
        float inpacc_27107;
        int64_t inpacc_27113;
        float inpacc_27115;
        
        inpacc_27113 = 0;
        inpacc_27115 = 0.0F;
        for (int64_t i_28327 = 0; i_28327 < res_27069; i_28327++) {
            int64_t x_28397 = ((int64_t *) mem_28708)[i_28327];
            int64_t i_p_o_28399 = add64(-1, i_28327);
            int64_t rot_i_28400 = smod64(i_p_o_28399, res_27069);
            int64_t x_28401 = ((int64_t *) mem_28708)[rot_i_28400];
            bool res_28402 = x_28397 == x_28401;
            bool res_28403 = !res_28402;
            int64_t res_27149;
            
            if (res_28403) {
                res_27149 = 1;
            } else {
                int64_t res_27150 = add64(1, inpacc_27113);
                
                res_27149 = res_27150;
            }
            
            int64_t res_27164;
            
            if (res_28403) {
                res_27164 = 1;
            } else {
                int64_t res_27165 = add64(1, inpacc_27113);
                
                res_27164 = res_27165;
            }
            
            int64_t res_27166 = sub64(res_27164, 1);
            bool x_28418 = sle64(0, x_28397);
            bool y_28419 = slt64(x_28397, flat_dim_26795);
            bool bounds_check_28420 = x_28418 && y_28419;
            bool index_certs_28421;
            
            if (!bounds_check_28420) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", x_28397,
                              "] out of bounds for array of shape [",
                              flat_dim_26795, "].",
                              "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:102:1-147:20\n");
                if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                    return 1;
                return 1;
            }
            
            int64_t new_index_28422 = squot64(x_28397, n_26622);
            int64_t binop_y_28423 = n_26622 * new_index_28422;
            int64_t new_index_28424 = x_28397 - binop_y_28423;
            float lifted_0_get_arg_28425 =
                  ((float *) mem_28644)[new_index_28422 * n_26622 +
                                        new_index_28424];
            float lifted_0_get_arg_28426 =
                  ((float *) mem_28514)[new_index_28424];
            float lifted_0_get_arg_28427 =
                  ((float *) mem_28516)[new_index_28424];
            int64_t lifted_0_get_arg_28428 =
                    ((int64_t *) mem_28518)[new_index_28424];
            float lifted_0_get_arg_28429 =
                  ((float *) mem_28520)[new_index_28424];
            float x_28430 = res_27049 / lifted_0_get_arg_28429;
            float ceil_arg_28431 = x_28430 - 1.0F;
            float res_28432;
            
            res_28432 = futrts_ceil32(ceil_arg_28431);
            
            int64_t res_28433 = fptosi_f32_i64(res_28432);
            int64_t max_arg_28434 = sub64(lifted_0_get_arg_28428, res_28433);
            int64_t res_28435 = smax64(0, max_arg_28434);
            bool cond_28436 = res_28435 == 0;
            float res_28437;
            
            if (cond_28436) {
                res_28437 = 0.0F;
            } else {
                float res_28438;
                
                res_28438 = futrts_ceil32(x_28430);
                
                float start_28439 = lifted_0_get_arg_28429 * res_28438;
                float res_28440;
                
                res_28440 = futrts_ceil32(ceil_arg_28431);
                
                int64_t res_28441 = fptosi_f32_i64(res_28440);
                int64_t max_arg_28442 = sub64(lifted_0_get_arg_28428,
                                              res_28441);
                int64_t res_28443 = smax64(0, max_arg_28442);
                int64_t sizze_28444 = sub64(res_28443, 1);
                bool cond_28445 = res_27166 == 0;
                float res_28446;
                
                if (cond_28445) {
                    res_28446 = 1.0F;
                } else {
                    res_28446 = 0.0F;
                }
                
                bool cond_28447 = slt64(0, res_27166);
                float res_28448;
                
                if (cond_28447) {
                    float y_28449 = lifted_0_get_arg_28426 *
                          lifted_0_get_arg_28429;
                    float res_28450 = res_28446 - y_28449;
                    
                    res_28448 = res_28450;
                } else {
                    res_28448 = res_28446;
                }
                
                bool cond_28451 = res_27166 == sizze_28444;
                float res_28452;
                
                if (cond_28451) {
                    float res_28453 = res_28448 - 1.0F;
                    
                    res_28452 = res_28453;
                } else {
                    res_28452 = res_28448;
                }
                
                float res_28454 = lifted_0_get_arg_28427 * res_28452;
                float res_28455 = sitofp_i64_f32(res_27166);
                float y_28456 = lifted_0_get_arg_28429 * res_28455;
                float bondprice_arg_28457 = start_28439 + y_28456;
                float y_28458 = bondprice_arg_28457 - res_27049;
                float negate_arg_28459 = a_26630 * y_28458;
                float exp_arg_28460 = 0.0F - negate_arg_28459;
                float res_28461 = fpow32(2.7182817F, exp_arg_28460);
                float x_28462 = 1.0F - res_28461;
                float B_28463 = x_28462 / a_26630;
                float x_28464 = B_28463 - bondprice_arg_28457;
                float x_28465 = res_27049 + x_28464;
                float x_28471 = y_26654 * x_28465;
                float A1_28472 = x_28471 / x_26650;
                float y_28473 = fpow32(B_28463, 2.0F);
                float x_28474 = x_26652 * y_28473;
                float A2_28476 = x_28474 / y_26655;
                float exp_arg_28477 = A1_28472 - A2_28476;
                float res_28478 = fpow32(2.7182817F, exp_arg_28477);
                float negate_arg_28479 = lifted_0_get_arg_28425 * B_28463;
                float exp_arg_28480 = 0.0F - negate_arg_28479;
                float res_28481 = fpow32(2.7182817F, exp_arg_28480);
                float res_28482 = res_28478 * res_28481;
                float x_28483 = res_28454 * res_28482;
                float res_28484 = lifted_0_get_arg_28427 * x_28483;
                
                res_28437 = res_28484;
            }
            
            float res_27243;
            
            if (res_28403) {
                res_27243 = res_28437;
            } else {
                float res_27244 = inpacc_27115 + res_28437;
                
                res_27243 = res_27244;
            }
            
            float res_27246;
            
            if (res_28403) {
                res_27246 = res_28437;
            } else {
                float res_27247 = inpacc_27115 + res_28437;
                
                res_27246 = res_27247;
            }
            ((float *) mem_28722)[i_28327] = res_27243;
            ((bool *) mem_28724)[i_28327] = res_28403;
            
            int64_t inpacc_tmp_28835 = res_27149;
            float inpacc_tmp_28836 = res_27246;
            
            inpacc_27113 = inpacc_tmp_28835;
            inpacc_27115 = inpacc_tmp_28836;
        }
        inpacc_27105 = inpacc_27113;
        inpacc_27107 = inpacc_27115;
        if (mem_28750_cached_sizze_28866 < (size_t) bytes_28693) {
            mem_28750 = realloc(mem_28750, bytes_28693);
            mem_28750_cached_sizze_28866 = bytes_28693;
        }
        
        int64_t discard_28336;
        int64_t scanacc_28332 = 0;
        
        for (int64_t i_28334 = 0; i_28334 < res_27069; i_28334++) {
            int64_t i_p_o_28495 = add64(1, i_28334);
            int64_t rot_i_28496 = smod64(i_p_o_28495, res_27069);
            bool x_27260 = ((bool *) mem_28724)[rot_i_28496];
            int64_t res_27261 = btoi_bool_i64(x_27260);
            int64_t res_27259 = add64(res_27261, scanacc_28332);
            
            ((int64_t *) mem_28750)[i_28334] = res_27259;
            
            int64_t scanacc_tmp_28839 = res_27259;
            
            scanacc_28332 = scanacc_tmp_28839;
        }
        discard_28336 = scanacc_28332;
        
        bool cond_27262 = slt64(0, res_27069);
        int64_t num_segments_27263;
        
        if (cond_27262) {
            int64_t i_27264 = sub64(res_27069, 1);
            bool x_27265 = sle64(0, i_27264);
            bool y_27266 = slt64(i_27264, res_27069);
            bool bounds_check_27267 = x_27265 && y_27266;
            bool index_certs_27268;
            
            if (!bounds_check_27267) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27264,
                              "] out of bounds for array of shape [", res_27069,
                              "].",
                              "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva.fut:134:38-97\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:127:44-137:50\n   #8  cva.fut:102:1-147:20\n");
                if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                    return 1;
                return 1;
            }
            
            int64_t res_27269 = ((int64_t *) mem_28750)[i_27264];
            
            num_segments_27263 = res_27269;
        } else {
            num_segments_27263 = 0;
        }
        
        bool bounds_invalid_upwards_27270 = slt64(num_segments_27263, 0);
        bool valid_27271 = !bounds_invalid_upwards_27270;
        bool range_valid_c_27272;
        
        if (!valid_27271) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", num_segments_27263,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:102:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                return 1;
            return 1;
        }
        
        int64_t bytes_28763 = 4 * num_segments_27263;
        
        if (mem_28764_cached_sizze_28867 < (size_t) bytes_28763) {
            mem_28764 = realloc(mem_28764, bytes_28763);
            mem_28764_cached_sizze_28867 = bytes_28763;
        }
        for (int64_t i_28841 = 0; i_28841 < num_segments_27263; i_28841++) {
            ((float *) mem_28764)[i_28841] = 0.0F;
        }
        for (int64_t write_iter_28337 = 0; write_iter_28337 < res_27069;
             write_iter_28337++) {
            int64_t write_iv_28339 = ((int64_t *) mem_28750)[write_iter_28337];
            int64_t i_p_o_28498 = add64(1, write_iter_28337);
            int64_t rot_i_28499 = smod64(i_p_o_28498, res_27069);
            bool write_iv_28340 = ((bool *) mem_28724)[rot_i_28499];
            int64_t res_27278;
            
            if (write_iv_28340) {
                int64_t res_27279 = sub64(write_iv_28339, 1);
                
                res_27278 = res_27279;
            } else {
                res_27278 = -1;
            }
            
            bool less_than_zzero_28342 = slt64(res_27278, 0);
            bool greater_than_sizze_28343 = sle64(num_segments_27263,
                                                  res_27278);
            bool outside_bounds_dim_28344 = less_than_zzero_28342 ||
                 greater_than_sizze_28343;
            
            if (!outside_bounds_dim_28344) {
                memmove(mem_28764 + res_27278 * 4, mem_28722 +
                        write_iter_28337 * 4, (int32_t) sizeof(float));
            }
        }
        
        bool dim_match_27280 = flat_dim_26795 == num_segments_27263;
        bool empty_or_match_cert_27281;
        
        if (!dim_match_27280) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Value of (core language) shape (",
                                   num_segments_27263,
                                   ") cannot match shape of type `[",
                                   flat_dim_26795, "]b`.",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva.fut:134:38-97\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:127:44-137:50\n   #5  cva.fut:102:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
                return 1;
            return 1;
        }
        
        float res_27283;
        float redout_28350 = 0.0F;
        
        for (int64_t i_28351 = 0; i_28351 < paths_26625; i_28351++) {
            int64_t binop_x_28502 = n_26622 * i_28351;
            float res_27288;
            float redout_28348 = 0.0F;
            
            for (int64_t i_28349 = 0; i_28349 < n_26622; i_28349++) {
                int64_t new_index_28503 = i_28349 + binop_x_28502;
                float x_27292 = ((float *) mem_28764)[new_index_28503];
                float res_27291 = x_27292 + redout_28348;
                float redout_tmp_28844 = res_27291;
                
                redout_28348 = redout_tmp_28844;
            }
            res_27288 = redout_28348;
            
            float res_27293 = fmax32(0.0F, res_27288);
            float res_27286 = res_27293 + redout_28350;
            float redout_tmp_28843 = res_27286;
            
            redout_28350 = redout_tmp_28843;
        }
        res_27283 = redout_28350;
        
        float res_27294 = res_27283 / res_27039;
        float negate_arg_27295 = a_26630 * res_27049;
        float exp_arg_27296 = 0.0F - negate_arg_27295;
        float res_27297 = fpow32(2.7182817F, exp_arg_27296);
        float x_27298 = 1.0F - res_27297;
        float B_27299 = x_27298 / a_26630;
        float x_27300 = B_27299 - res_27049;
        float x_27301 = y_26654 * x_27300;
        float A1_27302 = x_27301 / x_26650;
        float y_27303 = fpow32(B_27299, 2.0F);
        float x_27304 = x_26652 * y_27303;
        float A2_27305 = x_27304 / y_26655;
        float exp_arg_27306 = A1_27302 - A2_27305;
        float res_27307 = fpow32(2.7182817F, exp_arg_27306);
        float negate_arg_27308 = 5.0e-2F * B_27299;
        float exp_arg_27309 = 0.0F - negate_arg_27308;
        float res_27310 = fpow32(2.7182817F, exp_arg_27309);
        float res_27311 = res_27307 * res_27310;
        float res_27312 = res_27294 * res_27311;
        float res_27045 = res_27312 + redout_28353;
        
        memmove(mem_28627 + i_28355 * flat_dim_26795 * 4, mem_28764 + 0,
                paths_26625 * n_26622 * (int32_t) sizeof(float));
        
        float redout_tmp_28823 = res_27045;
        
        redout_28353 = redout_tmp_28823;
    }
    res_27041 = redout_28353;
    
    float CVA_27315 = 6.0e-3F * res_27041;
    struct memblock mem_28791;
    
    mem_28791.references = NULL;
    if (memblock_alloc(ctx, &mem_28791, bytes_28624, "mem_28791")) {
        err = 1;
        goto cleanup;
    }
    memmove(mem_28791.mem + 0, mem_28627 + 0, steps_26626 * paths_26625 *
            n_26622 * (int32_t) sizeof(float));
    out_arrsizze_28810 = steps_26626;
    out_arrsizze_28811 = paths_26625;
    out_arrsizze_28812 = n_26622;
    if (memblock_set(ctx, &out_mem_28809, &mem_28791, "mem_28791") != 0)
        return 1;
    scalar_out_28808 = CVA_27315;
    *out_scalar_out_28845 = scalar_out_28808;
    (*out_mem_p_28846).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_28846, &out_mem_28809, "out_mem_28809") !=
        0)
        return 1;
    *out_out_arrsizze_28847 = out_arrsizze_28810;
    *out_out_arrsizze_28848 = out_arrsizze_28811;
    *out_out_arrsizze_28849 = out_arrsizze_28812;
    if (memblock_unref(ctx, &mem_28791, "mem_28791") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_28809, "out_mem_28809") != 0)
        return 1;
    
  cleanup:
    { }
    free(mem_28514);
    free(mem_28516);
    free(mem_28518);
    free(mem_28520);
    free(mem_28571);
    free(mem_28583);
    free(mem_28597);
    free(mem_28627);
    free(mem_28644);
    free(mem_28656);
    free(mem_28666);
    free(mem_28668);
    free(mem_28694);
    free(mem_28708);
    free(mem_28722);
    free(mem_28724);
    free(mem_28750);
    free(mem_28764);
    return err;
}
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_28868,
                       int64_t paths_27317, int64_t steps_27318)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_28511_cached_sizze_28869 = 0;
    char *mem_28511 = NULL;
    size_t mem_28513_cached_sizze_28870 = 0;
    char *mem_28513 = NULL;
    size_t mem_28515_cached_sizze_28871 = 0;
    char *mem_28515 = NULL;
    size_t mem_28517_cached_sizze_28872 = 0;
    char *mem_28517 = NULL;
    size_t mem_28519_cached_sizze_28873 = 0;
    char *mem_28519 = NULL;
    size_t mem_28521_cached_sizze_28874 = 0;
    char *mem_28521 = NULL;
    size_t mem_28523_cached_sizze_28875 = 0;
    char *mem_28523 = NULL;
    size_t mem_28582_cached_sizze_28876 = 0;
    char *mem_28582 = NULL;
    size_t mem_28594_cached_sizze_28877 = 0;
    char *mem_28594 = NULL;
    size_t mem_28608_cached_sizze_28878 = 0;
    char *mem_28608 = NULL;
    size_t mem_28637_cached_sizze_28879 = 0;
    char *mem_28637 = NULL;
    size_t mem_28652_cached_sizze_28880 = 0;
    char *mem_28652 = NULL;
    size_t mem_28662_cached_sizze_28881 = 0;
    char *mem_28662 = NULL;
    size_t mem_28664_cached_sizze_28882 = 0;
    char *mem_28664 = NULL;
    size_t mem_28690_cached_sizze_28883 = 0;
    char *mem_28690 = NULL;
    size_t mem_28704_cached_sizze_28884 = 0;
    char *mem_28704 = NULL;
    size_t mem_28718_cached_sizze_28885 = 0;
    char *mem_28718 = NULL;
    size_t mem_28720_cached_sizze_28886 = 0;
    char *mem_28720 = NULL;
    size_t mem_28746_cached_sizze_28887 = 0;
    char *mem_28746 = NULL;
    size_t mem_28760_cached_sizze_28888 = 0;
    char *mem_28760 = NULL;
    float scalar_out_28808;
    
    if (mem_28511_cached_sizze_28869 < (size_t) 180) {
        mem_28511 = realloc(mem_28511, 180);
        mem_28511_cached_sizze_28869 = 180;
    }
    
    struct memblock testzistatic_array_28809 = ctx->testzistatic_array_28809;
    
    memmove(mem_28511 + 0, testzistatic_array_28809.mem + 0, 45 *
            (int32_t) sizeof(float));
    if (mem_28513_cached_sizze_28870 < (size_t) 360) {
        mem_28513 = realloc(mem_28513, 360);
        mem_28513_cached_sizze_28870 = 360;
    }
    
    struct memblock testzistatic_array_28810 = ctx->testzistatic_array_28810;
    
    memmove(mem_28513 + 0, testzistatic_array_28810.mem + 0, 45 *
            (int32_t) sizeof(int64_t));
    if (mem_28515_cached_sizze_28871 < (size_t) 180) {
        mem_28515 = realloc(mem_28515, 180);
        mem_28515_cached_sizze_28871 = 180;
    }
    
    struct memblock testzistatic_array_28811 = ctx->testzistatic_array_28811;
    
    memmove(mem_28515 + 0, testzistatic_array_28811.mem + 0, 45 *
            (int32_t) sizeof(float));
    
    float res_27322;
    float redout_28227 = -INFINITY;
    
    for (int32_t i_28363 = 0; i_28363 < 45; i_28363++) {
        int64_t i_28228 = sext_i32_i64(i_28363);
        float x_27326 = ((float *) mem_28515)[i_28228];
        int64_t x_27327 = ((int64_t *) mem_28513)[i_28228];
        float res_27328 = sitofp_i64_f32(x_27327);
        float res_27329 = x_27326 * res_27328;
        float res_27325 = fmax32(res_27329, redout_28227);
        float redout_tmp_28812 = res_27325;
        
        redout_28227 = redout_tmp_28812;
    }
    res_27322 = redout_28227;
    
    float res_27330 = sitofp_i64_f32(steps_27318);
    float dt_27331 = res_27322 / res_27330;
    
    if (mem_28517_cached_sizze_28872 < (size_t) 180) {
        mem_28517 = realloc(mem_28517, 180);
        mem_28517_cached_sizze_28872 = 180;
    }
    if (mem_28519_cached_sizze_28873 < (size_t) 180) {
        mem_28519 = realloc(mem_28519, 180);
        mem_28519_cached_sizze_28873 = 180;
    }
    if (mem_28521_cached_sizze_28874 < (size_t) 360) {
        mem_28521 = realloc(mem_28521, 360);
        mem_28521_cached_sizze_28874 = 360;
    }
    if (mem_28523_cached_sizze_28875 < (size_t) 180) {
        mem_28523 = realloc(mem_28523, 180);
        mem_28523_cached_sizze_28875 = 180;
    }
    for (int32_t i_28371 = 0; i_28371 < 45; i_28371++) {
        int64_t i_28239 = sext_i32_i64(i_28371);
        bool x_27338 = sle64(0, i_28239);
        bool y_27339 = slt64(i_28239, 45);
        bool bounds_check_27340 = x_27338 && y_27339;
        bool index_certs_27341;
        
        if (!bounds_check_27340) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", i_28239,
                                   "] out of bounds for array of shape [", 45,
                                   "].",
                                   "-> #0  cva.fut:108:15-26\n   #1  cva.fut:107:17-111:85\n   #2  cva.fut:156:3-158:129\n   #3  cva.fut:155:1-158:137\n");
            return 1;
        }
        
        float res_27342 = ((float *) mem_28515)[i_28239];
        int64_t res_27343 = ((int64_t *) mem_28513)[i_28239];
        bool bounds_invalid_upwards_27345 = slt64(res_27343, 1);
        bool valid_27346 = !bounds_invalid_upwards_27345;
        bool range_valid_c_27347;
        
        if (!valid_27346) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_27343,
                          " is invalid.",
                          "-> #0  cva.fut:55:29-48\n   #1  cva.fut:96:25-65\n   #2  cva.fut:111:16-62\n   #3  cva.fut:107:17-111:85\n   #4  cva.fut:156:3-158:129\n   #5  cva.fut:155:1-158:137\n");
            return 1;
        }
        
        bool y_27349 = slt64(0, res_27343);
        bool index_certs_27350;
        
        if (!y_27349) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_27343, "].",
                                   "-> #0  cva.fut:97:47-70\n   #1  cva.fut:111:16-62\n   #2  cva.fut:107:17-111:85\n   #3  cva.fut:156:3-158:129\n   #4  cva.fut:155:1-158:137\n");
            return 1;
        }
        
        float binop_y_27351 = sitofp_i64_f32(res_27343);
        float index_primexp_27352 = res_27342 * binop_y_27351;
        float negate_arg_27353 = 1.0e-2F * index_primexp_27352;
        float exp_arg_27354 = 0.0F - negate_arg_27353;
        float res_27355 = fpow32(2.7182817F, exp_arg_27354);
        float x_27356 = 1.0F - res_27355;
        float B_27357 = x_27356 / 1.0e-2F;
        float x_27358 = B_27357 - index_primexp_27352;
        float x_27359 = 4.4999997e-6F * x_27358;
        float A1_27360 = x_27359 / 1.0e-4F;
        float y_27361 = fpow32(B_27357, 2.0F);
        float x_27362 = 1.0000001e-6F * y_27361;
        float A2_27363 = x_27362 / 4.0e-2F;
        float exp_arg_27364 = A1_27360 - A2_27363;
        float res_27365 = fpow32(2.7182817F, exp_arg_27364);
        float negate_arg_27366 = 5.0e-2F * B_27357;
        float exp_arg_27367 = 0.0F - negate_arg_27366;
        float res_27368 = fpow32(2.7182817F, exp_arg_27367);
        float res_27369 = res_27365 * res_27368;
        float res_27370;
        float redout_28229 = 0.0F;
        
        for (int64_t i_28230 = 0; i_28230 < res_27343; i_28230++) {
            int64_t index_primexp_28365 = add64(1, i_28230);
            float res_27375 = sitofp_i64_f32(index_primexp_28365);
            float res_27376 = res_27342 * res_27375;
            float negate_arg_27377 = 1.0e-2F * res_27376;
            float exp_arg_27378 = 0.0F - negate_arg_27377;
            float res_27379 = fpow32(2.7182817F, exp_arg_27378);
            float x_27380 = 1.0F - res_27379;
            float B_27381 = x_27380 / 1.0e-2F;
            float x_27382 = B_27381 - res_27376;
            float x_27383 = 4.4999997e-6F * x_27382;
            float A1_27384 = x_27383 / 1.0e-4F;
            float y_27385 = fpow32(B_27381, 2.0F);
            float x_27386 = 1.0000001e-6F * y_27385;
            float A2_27387 = x_27386 / 4.0e-2F;
            float exp_arg_27388 = A1_27384 - A2_27387;
            float res_27389 = fpow32(2.7182817F, exp_arg_27388);
            float negate_arg_27390 = 5.0e-2F * B_27381;
            float exp_arg_27391 = 0.0F - negate_arg_27390;
            float res_27392 = fpow32(2.7182817F, exp_arg_27391);
            float res_27393 = res_27389 * res_27392;
            float res_27373 = res_27393 + redout_28229;
            float redout_tmp_28817 = res_27373;
            
            redout_28229 = redout_tmp_28817;
        }
        res_27370 = redout_28229;
        
        float x_27394 = 1.0F - res_27369;
        float y_27395 = res_27342 * res_27370;
        float res_27396 = x_27394 / y_27395;
        
        ((float *) mem_28517)[i_28239] = res_27396;
        memmove(mem_28519 + i_28239 * 4, mem_28511 + i_28239 * 4,
                (int32_t) sizeof(float));
        memmove(mem_28521 + i_28239 * 8, mem_28513 + i_28239 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_28523 + i_28239 * 4, mem_28515 + i_28239 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_27397 = res_27330 / res_27322;
    bool bounds_invalid_upwards_27398 = slt64(steps_27318, 1);
    bool valid_27399 = !bounds_invalid_upwards_27398;
    bool range_valid_c_27400;
    
    if (!valid_27399) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_27318,
                               " is invalid.",
                               "-> #0  cva.fut:61:56-67\n   #1  cva.fut:113:17-44\n   #2  cva.fut:156:3-158:129\n   #3  cva.fut:155:1-158:137\n");
        return 1;
    }
    
    bool bounds_invalid_upwards_27402 = slt64(paths_27317, 0);
    bool valid_27403 = !bounds_invalid_upwards_27402;
    bool range_valid_c_27404;
    
    if (!valid_27403) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_27317,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:116:19-49\n   #4  cva.fut:156:3-158:129\n   #5  cva.fut:155:1-158:137\n");
        return 1;
    }
    
    int64_t upper_bound_27407 = sub64(steps_27318, 1);
    float res_27408;
    
    res_27408 = futrts_sqrt32(dt_27331);
    
    int64_t binop_x_28581 = paths_27317 * steps_27318;
    int64_t bytes_28580 = 4 * binop_x_28581;
    
    if (mem_28582_cached_sizze_28876 < (size_t) bytes_28580) {
        mem_28582 = realloc(mem_28582, bytes_28580);
        mem_28582_cached_sizze_28876 = bytes_28580;
    }
    
    int64_t bytes_28593 = 4 * steps_27318;
    
    if (mem_28594_cached_sizze_28877 < (size_t) bytes_28593) {
        mem_28594 = realloc(mem_28594, bytes_28593);
        mem_28594_cached_sizze_28877 = bytes_28593;
    }
    if (mem_28608_cached_sizze_28878 < (size_t) bytes_28593) {
        mem_28608 = realloc(mem_28608, bytes_28593);
        mem_28608_cached_sizze_28878 = bytes_28593;
    }
    for (int64_t i_28818 = 0; i_28818 < steps_27318; i_28818++) {
        ((float *) mem_28608)[i_28818] = 5.0e-2F;
    }
    for (int64_t i_28250 = 0; i_28250 < paths_27317; i_28250++) {
        int32_t res_27411 = sext_i64_i32(i_28250);
        int32_t x_27412 = lshr32(res_27411, 16);
        int32_t x_27413 = res_27411 ^ x_27412;
        int32_t x_27414 = mul32(73244475, x_27413);
        int32_t x_27415 = lshr32(x_27414, 16);
        int32_t x_27416 = x_27414 ^ x_27415;
        int32_t x_27417 = mul32(73244475, x_27416);
        int32_t x_27418 = lshr32(x_27417, 16);
        int32_t x_27419 = x_27417 ^ x_27418;
        int32_t unsign_arg_27420 = 777822902 ^ x_27419;
        int32_t unsign_arg_27421 = mul32(48271, unsign_arg_27420);
        int32_t unsign_arg_27422 = umod32(unsign_arg_27421, 2147483647);
        
        for (int64_t i_28246 = 0; i_28246 < steps_27318; i_28246++) {
            int32_t res_27425 = sext_i64_i32(i_28246);
            int32_t x_27426 = lshr32(res_27425, 16);
            int32_t x_27427 = res_27425 ^ x_27426;
            int32_t x_27428 = mul32(73244475, x_27427);
            int32_t x_27429 = lshr32(x_27428, 16);
            int32_t x_27430 = x_27428 ^ x_27429;
            int32_t x_27431 = mul32(73244475, x_27430);
            int32_t x_27432 = lshr32(x_27431, 16);
            int32_t x_27433 = x_27431 ^ x_27432;
            int32_t unsign_arg_27434 = unsign_arg_27422 ^ x_27433;
            int32_t unsign_arg_27435 = mul32(48271, unsign_arg_27434);
            int32_t unsign_arg_27436 = umod32(unsign_arg_27435, 2147483647);
            int32_t unsign_arg_27437 = mul32(48271, unsign_arg_27436);
            int32_t unsign_arg_27438 = umod32(unsign_arg_27437, 2147483647);
            float res_27439 = uitofp_i32_f32(unsign_arg_27436);
            float res_27440 = res_27439 / 2.1474836e9F;
            float res_27441 = uitofp_i32_f32(unsign_arg_27438);
            float res_27442 = res_27441 / 2.1474836e9F;
            float res_27443;
            
            res_27443 = futrts_log32(res_27440);
            
            float res_27444 = -2.0F * res_27443;
            float res_27445;
            
            res_27445 = futrts_sqrt32(res_27444);
            
            float res_27446 = 6.2831855F * res_27442;
            float res_27447;
            
            res_27447 = futrts_cos32(res_27446);
            
            float res_27448 = res_27445 * res_27447;
            
            ((float *) mem_28594)[i_28246] = res_27448;
        }
        memmove(mem_28582 + i_28250 * steps_27318 * 4, mem_28608 + 0,
                steps_27318 * (int32_t) sizeof(float));
        for (int64_t i_27451 = 0; i_27451 < upper_bound_27407; i_27451++) {
            bool y_27453 = slt64(i_27451, steps_27318);
            bool index_certs_27454;
            
            if (!y_27453) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27451,
                              "] out of bounds for array of shape [",
                              steps_27318, "].",
                              "-> #0  cva.fut:72:97-104\n   #1  cva.fut:124:32-62\n   #2  cva.fut:124:22-69\n   #3  cva.fut:156:3-158:129\n   #4  cva.fut:155:1-158:137\n");
                return 1;
            }
            
            float shortstep_arg_27455 = ((float *) mem_28594)[i_27451];
            float shortstep_arg_27456 = ((float *) mem_28582)[i_28250 *
                                                              steps_27318 +
                                                              i_27451];
            float y_27457 = 5.0e-2F - shortstep_arg_27456;
            float x_27458 = 1.0e-2F * y_27457;
            float x_27459 = dt_27331 * x_27458;
            float x_27460 = res_27408 * shortstep_arg_27455;
            float y_27461 = 1.0e-3F * x_27460;
            float delta_r_27462 = x_27459 + y_27461;
            float res_27463 = shortstep_arg_27456 + delta_r_27462;
            int64_t i_27464 = add64(1, i_27451);
            bool x_27465 = sle64(0, i_27464);
            bool y_27466 = slt64(i_27464, steps_27318);
            bool bounds_check_27467 = x_27465 && y_27466;
            bool index_certs_27468;
            
            if (!bounds_check_27467) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27464,
                              "] out of bounds for array of shape [",
                              steps_27318, "].",
                              "-> #0  cva.fut:72:58-105\n   #1  cva.fut:124:32-62\n   #2  cva.fut:124:22-69\n   #3  cva.fut:156:3-158:129\n   #4  cva.fut:155:1-158:137\n");
                return 1;
            }
            ((float *) mem_28582)[i_28250 * steps_27318 + i_27464] = res_27463;
        }
    }
    
    int64_t flat_dim_27472 = 45 * paths_27317;
    float res_27476 = sitofp_i64_f32(paths_27317);
    int64_t binop_x_28636 = 45 * paths_27317;
    int64_t bytes_28635 = 4 * binop_x_28636;
    
    if (mem_28637_cached_sizze_28879 < (size_t) bytes_28635) {
        mem_28637 = realloc(mem_28637, bytes_28635);
        mem_28637_cached_sizze_28879 = bytes_28635;
    }
    if (mem_28652_cached_sizze_28880 < (size_t) 180) {
        mem_28652 = realloc(mem_28652, 180);
        mem_28652_cached_sizze_28880 = 180;
    }
    
    int64_t bytes_28661 = 8 * flat_dim_27472;
    
    if (mem_28662_cached_sizze_28881 < (size_t) bytes_28661) {
        mem_28662 = realloc(mem_28662, bytes_28661);
        mem_28662_cached_sizze_28881 = bytes_28661;
    }
    if (mem_28664_cached_sizze_28882 < (size_t) bytes_28661) {
        mem_28664 = realloc(mem_28664, bytes_28661);
        mem_28664_cached_sizze_28882 = bytes_28661;
    }
    
    float res_27478;
    float redout_28352 = 0.0F;
    
    for (int64_t i_28353 = 0; i_28353 < steps_27318; i_28353++) {
        int64_t index_primexp_28498 = add64(1, i_28353);
        float res_27484 = sitofp_i64_f32(index_primexp_28498);
        float res_27485 = res_27484 / sims_per_year_27397;
        
        for (int64_t i_28254 = 0; i_28254 < paths_27317; i_28254++) {
            float x_27487 = ((float *) mem_28582)[i_28254 * steps_27318 +
                                                  i_28353];
            
            for (int64_t i_28824 = 0; i_28824 < 45; i_28824++) {
                ((float *) mem_28652)[i_28824] = x_27487;
            }
            memmove(mem_28637 + i_28254 * 45 * 4, mem_28652 + 0, 45 *
                    (int32_t) sizeof(float));
        }
        
        int64_t discard_28264;
        int64_t scanacc_28258 = 0;
        
        for (int64_t i_28261 = 0; i_28261 < flat_dim_27472; i_28261++) {
            int64_t binop_x_28377 = squot64(i_28261, 45);
            int64_t binop_y_28378 = 45 * binop_x_28377;
            int64_t new_index_28379 = i_28261 - binop_y_28378;
            int64_t x_27494 = ((int64_t *) mem_28521)[new_index_28379];
            float x_27495 = ((float *) mem_28523)[new_index_28379];
            float x_27496 = res_27485 / x_27495;
            float ceil_arg_27497 = x_27496 - 1.0F;
            float res_27498;
            
            res_27498 = futrts_ceil32(ceil_arg_27497);
            
            int64_t res_27499 = fptosi_f32_i64(res_27498);
            int64_t max_arg_27500 = sub64(x_27494, res_27499);
            int64_t res_27501 = smax64(0, max_arg_27500);
            bool cond_27502 = res_27501 == 0;
            int64_t res_27503;
            
            if (cond_27502) {
                res_27503 = 1;
            } else {
                res_27503 = res_27501;
            }
            
            int64_t res_27493 = add64(res_27503, scanacc_28258);
            
            ((int64_t *) mem_28662)[i_28261] = res_27493;
            ((int64_t *) mem_28664)[i_28261] = res_27503;
            
            int64_t scanacc_tmp_28825 = res_27493;
            
            scanacc_28258 = scanacc_tmp_28825;
        }
        discard_28264 = scanacc_28258;
        
        int64_t res_27505;
        int64_t redout_28265 = 0;
        
        for (int64_t i_28266 = 0; i_28266 < flat_dim_27472; i_28266++) {
            int64_t x_27509 = ((int64_t *) mem_28664)[i_28266];
            int64_t res_27508 = add64(x_27509, redout_28265);
            int64_t redout_tmp_28828 = res_27508;
            
            redout_28265 = redout_tmp_28828;
        }
        res_27505 = redout_28265;
        
        bool bounds_invalid_upwards_27510 = slt64(res_27505, 0);
        bool valid_27511 = !bounds_invalid_upwards_27510;
        bool range_valid_c_27512;
        
        if (!valid_27511) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", res_27505,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:156:3-158:129\n   #10 cva.fut:155:1-158:137\n");
            return 1;
        }
        
        int64_t bytes_28689 = 8 * res_27505;
        
        if (mem_28690_cached_sizze_28883 < (size_t) bytes_28689) {
            mem_28690 = realloc(mem_28690, bytes_28689);
            mem_28690_cached_sizze_28883 = bytes_28689;
        }
        for (int64_t i_28829 = 0; i_28829 < res_27505; i_28829++) {
            ((int64_t *) mem_28690)[i_28829] = 0;
        }
        for (int64_t iter_28267 = 0; iter_28267 < flat_dim_27472;
             iter_28267++) {
            int64_t i_p_o_28384 = add64(-1, iter_28267);
            int64_t rot_i_28385 = smod64(i_p_o_28384, flat_dim_27472);
            int64_t pixel_28270 = ((int64_t *) mem_28662)[rot_i_28385];
            bool cond_27520 = iter_28267 == 0;
            int64_t res_27521;
            
            if (cond_27520) {
                res_27521 = 0;
            } else {
                res_27521 = pixel_28270;
            }
            
            bool less_than_zzero_28271 = slt64(res_27521, 0);
            bool greater_than_sizze_28272 = sle64(res_27505, res_27521);
            bool outside_bounds_dim_28273 = less_than_zzero_28271 ||
                 greater_than_sizze_28272;
            
            if (!outside_bounds_dim_28273) {
                int64_t read_hist_28275 = ((int64_t *) mem_28690)[res_27521];
                int64_t res_27517 = smax64(iter_28267, read_hist_28275);
                
                ((int64_t *) mem_28690)[res_27521] = res_27517;
            }
        }
        if (mem_28704_cached_sizze_28884 < (size_t) bytes_28689) {
            mem_28704 = realloc(mem_28704, bytes_28689);
            mem_28704_cached_sizze_28884 = bytes_28689;
        }
        
        int64_t discard_28288;
        int64_t scanacc_28281 = 0;
        
        for (int64_t i_28284 = 0; i_28284 < res_27505; i_28284++) {
            int64_t x_27531 = ((int64_t *) mem_28690)[i_28284];
            bool res_27532 = slt64(0, x_27531);
            int64_t res_27529;
            
            if (res_27532) {
                res_27529 = x_27531;
            } else {
                int64_t res_27530 = add64(x_27531, scanacc_28281);
                
                res_27529 = res_27530;
            }
            ((int64_t *) mem_28704)[i_28284] = res_27529;
            
            int64_t scanacc_tmp_28831 = res_27529;
            
            scanacc_28281 = scanacc_tmp_28831;
        }
        discard_28288 = scanacc_28281;
        
        int64_t bytes_28717 = 4 * res_27505;
        
        if (mem_28718_cached_sizze_28885 < (size_t) bytes_28717) {
            mem_28718 = realloc(mem_28718, bytes_28717);
            mem_28718_cached_sizze_28885 = bytes_28717;
        }
        if (mem_28720_cached_sizze_28886 < (size_t) res_27505) {
            mem_28720 = realloc(mem_28720, res_27505);
            mem_28720_cached_sizze_28886 = res_27505;
        }
        
        int64_t inpacc_27541;
        float inpacc_27543;
        int64_t inpacc_27549;
        float inpacc_27551;
        
        inpacc_27549 = 0;
        inpacc_27551 = 0.0F;
        for (int64_t i_28327 = 0; i_28327 < res_27505; i_28327++) {
            int64_t x_28399 = ((int64_t *) mem_28704)[i_28327];
            int64_t i_p_o_28401 = add64(-1, i_28327);
            int64_t rot_i_28402 = smod64(i_p_o_28401, res_27505);
            int64_t x_28403 = ((int64_t *) mem_28704)[rot_i_28402];
            bool res_28404 = x_28399 == x_28403;
            bool res_28405 = !res_28404;
            int64_t res_27585;
            
            if (res_28405) {
                res_27585 = 1;
            } else {
                int64_t res_27586 = add64(1, inpacc_27549);
                
                res_27585 = res_27586;
            }
            
            int64_t res_27600;
            
            if (res_28405) {
                res_27600 = 1;
            } else {
                int64_t res_27601 = add64(1, inpacc_27549);
                
                res_27600 = res_27601;
            }
            
            int64_t res_27602 = sub64(res_27600, 1);
            bool x_28420 = sle64(0, x_28399);
            bool y_28421 = slt64(x_28399, flat_dim_27472);
            bool bounds_check_28422 = x_28420 && y_28421;
            bool index_certs_28423;
            
            if (!bounds_check_28422) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", x_28399,
                              "] out of bounds for array of shape [",
                              flat_dim_27472, "].",
                              "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:156:3-158:129\n   #10 cva.fut:155:1-158:137\n");
                return 1;
            }
            
            int64_t new_index_28424 = squot64(x_28399, 45);
            int64_t binop_y_28425 = 45 * new_index_28424;
            int64_t new_index_28426 = x_28399 - binop_y_28425;
            float lifted_0_get_arg_28427 =
                  ((float *) mem_28637)[new_index_28424 * 45 + new_index_28426];
            float lifted_0_get_arg_28428 =
                  ((float *) mem_28517)[new_index_28426];
            float lifted_0_get_arg_28429 =
                  ((float *) mem_28519)[new_index_28426];
            int64_t lifted_0_get_arg_28430 =
                    ((int64_t *) mem_28521)[new_index_28426];
            float lifted_0_get_arg_28431 =
                  ((float *) mem_28523)[new_index_28426];
            float x_28432 = res_27485 / lifted_0_get_arg_28431;
            float ceil_arg_28433 = x_28432 - 1.0F;
            float res_28434;
            
            res_28434 = futrts_ceil32(ceil_arg_28433);
            
            int64_t res_28435 = fptosi_f32_i64(res_28434);
            int64_t max_arg_28436 = sub64(lifted_0_get_arg_28430, res_28435);
            int64_t res_28437 = smax64(0, max_arg_28436);
            bool cond_28438 = res_28437 == 0;
            float res_28439;
            
            if (cond_28438) {
                res_28439 = 0.0F;
            } else {
                float res_28440;
                
                res_28440 = futrts_ceil32(x_28432);
                
                float start_28441 = lifted_0_get_arg_28431 * res_28440;
                float res_28442;
                
                res_28442 = futrts_ceil32(ceil_arg_28433);
                
                int64_t res_28443 = fptosi_f32_i64(res_28442);
                int64_t max_arg_28444 = sub64(lifted_0_get_arg_28430,
                                              res_28443);
                int64_t res_28445 = smax64(0, max_arg_28444);
                int64_t sizze_28446 = sub64(res_28445, 1);
                bool cond_28447 = res_27602 == 0;
                float res_28448;
                
                if (cond_28447) {
                    res_28448 = 1.0F;
                } else {
                    res_28448 = 0.0F;
                }
                
                bool cond_28449 = slt64(0, res_27602);
                float res_28450;
                
                if (cond_28449) {
                    float y_28451 = lifted_0_get_arg_28428 *
                          lifted_0_get_arg_28431;
                    float res_28452 = res_28448 - y_28451;
                    
                    res_28450 = res_28452;
                } else {
                    res_28450 = res_28448;
                }
                
                bool cond_28453 = res_27602 == sizze_28446;
                float res_28454;
                
                if (cond_28453) {
                    float res_28455 = res_28450 - 1.0F;
                    
                    res_28454 = res_28455;
                } else {
                    res_28454 = res_28450;
                }
                
                float res_28456 = lifted_0_get_arg_28429 * res_28454;
                float res_28457 = sitofp_i64_f32(res_27602);
                float y_28458 = lifted_0_get_arg_28431 * res_28457;
                float bondprice_arg_28459 = start_28441 + y_28458;
                float y_28460 = bondprice_arg_28459 - res_27485;
                float negate_arg_28461 = 1.0e-2F * y_28460;
                float exp_arg_28462 = 0.0F - negate_arg_28461;
                float res_28463 = fpow32(2.7182817F, exp_arg_28462);
                float x_28464 = 1.0F - res_28463;
                float B_28465 = x_28464 / 1.0e-2F;
                float x_28466 = B_28465 - bondprice_arg_28459;
                float x_28467 = res_27485 + x_28466;
                float x_28468 = 4.4999997e-6F * x_28467;
                float A1_28469 = x_28468 / 1.0e-4F;
                float y_28470 = fpow32(B_28465, 2.0F);
                float x_28471 = 1.0000001e-6F * y_28470;
                float A2_28472 = x_28471 / 4.0e-2F;
                float exp_arg_28473 = A1_28469 - A2_28472;
                float res_28474 = fpow32(2.7182817F, exp_arg_28473);
                float negate_arg_28475 = lifted_0_get_arg_28427 * B_28465;
                float exp_arg_28476 = 0.0F - negate_arg_28475;
                float res_28477 = fpow32(2.7182817F, exp_arg_28476);
                float res_28478 = res_28474 * res_28477;
                float x_28479 = res_28456 * res_28478;
                float res_28480 = lifted_0_get_arg_28429 * x_28479;
                
                res_28439 = res_28480;
            }
            
            float res_27673;
            
            if (res_28405) {
                res_27673 = res_28439;
            } else {
                float res_27674 = inpacc_27551 + res_28439;
                
                res_27673 = res_27674;
            }
            
            float res_27676;
            
            if (res_28405) {
                res_27676 = res_28439;
            } else {
                float res_27677 = inpacc_27551 + res_28439;
                
                res_27676 = res_27677;
            }
            ((float *) mem_28718)[i_28327] = res_27673;
            ((bool *) mem_28720)[i_28327] = res_28405;
            
            int64_t inpacc_tmp_28833 = res_27585;
            float inpacc_tmp_28834 = res_27676;
            
            inpacc_27549 = inpacc_tmp_28833;
            inpacc_27551 = inpacc_tmp_28834;
        }
        inpacc_27541 = inpacc_27549;
        inpacc_27543 = inpacc_27551;
        if (mem_28746_cached_sizze_28887 < (size_t) bytes_28689) {
            mem_28746 = realloc(mem_28746, bytes_28689);
            mem_28746_cached_sizze_28887 = bytes_28689;
        }
        
        int64_t discard_28336;
        int64_t scanacc_28332 = 0;
        
        for (int64_t i_28334 = 0; i_28334 < res_27505; i_28334++) {
            int64_t i_p_o_28491 = add64(1, i_28334);
            int64_t rot_i_28492 = smod64(i_p_o_28491, res_27505);
            bool x_27690 = ((bool *) mem_28720)[rot_i_28492];
            int64_t res_27691 = btoi_bool_i64(x_27690);
            int64_t res_27689 = add64(res_27691, scanacc_28332);
            
            ((int64_t *) mem_28746)[i_28334] = res_27689;
            
            int64_t scanacc_tmp_28837 = res_27689;
            
            scanacc_28332 = scanacc_tmp_28837;
        }
        discard_28336 = scanacc_28332;
        
        bool cond_27692 = slt64(0, res_27505);
        int64_t num_segments_27693;
        
        if (cond_27692) {
            int64_t i_27694 = sub64(res_27505, 1);
            bool x_27695 = sle64(0, i_27694);
            bool y_27696 = slt64(i_27694, res_27505);
            bool bounds_check_27697 = x_27695 && y_27696;
            bool index_certs_27698;
            
            if (!bounds_check_27697) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27694,
                              "] out of bounds for array of shape [", res_27505,
                              "].",
                              "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva.fut:134:38-97\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:127:44-137:50\n   #8  cva.fut:156:3-158:129\n   #9  cva.fut:155:1-158:137\n");
                return 1;
            }
            
            int64_t res_27699 = ((int64_t *) mem_28746)[i_27694];
            
            num_segments_27693 = res_27699;
        } else {
            num_segments_27693 = 0;
        }
        
        bool bounds_invalid_upwards_27700 = slt64(num_segments_27693, 0);
        bool valid_27701 = !bounds_invalid_upwards_27700;
        bool range_valid_c_27702;
        
        if (!valid_27701) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", num_segments_27693,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:156:3-158:129\n   #10 cva.fut:155:1-158:137\n");
            return 1;
        }
        
        int64_t bytes_28759 = 4 * num_segments_27693;
        
        if (mem_28760_cached_sizze_28888 < (size_t) bytes_28759) {
            mem_28760 = realloc(mem_28760, bytes_28759);
            mem_28760_cached_sizze_28888 = bytes_28759;
        }
        for (int64_t i_28839 = 0; i_28839 < num_segments_27693; i_28839++) {
            ((float *) mem_28760)[i_28839] = 0.0F;
        }
        for (int64_t write_iter_28337 = 0; write_iter_28337 < res_27505;
             write_iter_28337++) {
            int64_t write_iv_28339 = ((int64_t *) mem_28746)[write_iter_28337];
            int64_t i_p_o_28494 = add64(1, write_iter_28337);
            int64_t rot_i_28495 = smod64(i_p_o_28494, res_27505);
            bool write_iv_28340 = ((bool *) mem_28720)[rot_i_28495];
            int64_t res_27708;
            
            if (write_iv_28340) {
                int64_t res_27709 = sub64(write_iv_28339, 1);
                
                res_27708 = res_27709;
            } else {
                res_27708 = -1;
            }
            
            bool less_than_zzero_28342 = slt64(res_27708, 0);
            bool greater_than_sizze_28343 = sle64(num_segments_27693,
                                                  res_27708);
            bool outside_bounds_dim_28344 = less_than_zzero_28342 ||
                 greater_than_sizze_28343;
            
            if (!outside_bounds_dim_28344) {
                memmove(mem_28760 + res_27708 * 4, mem_28718 +
                        write_iter_28337 * 4, (int32_t) sizeof(float));
            }
        }
        
        bool dim_match_27710 = flat_dim_27472 == num_segments_27693;
        bool empty_or_match_cert_27711;
        
        if (!dim_match_27710) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Value of (core language) shape (",
                                   num_segments_27693,
                                   ") cannot match shape of type `[",
                                   flat_dim_27472, "]b`.",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva.fut:134:38-97\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:127:44-137:50\n   #5  cva.fut:156:3-158:129\n   #6  cva.fut:155:1-158:137\n");
            return 1;
        }
        
        float res_27713;
        float redout_28350 = 0.0F;
        
        for (int64_t i_28351 = 0; i_28351 < paths_27317; i_28351++) {
            int64_t binop_x_28499 = 45 * i_28351;
            float res_27718;
            float redout_28348 = 0.0F;
            
            for (int32_t i_28496 = 0; i_28496 < 45; i_28496++) {
                int64_t i_28349 = sext_i32_i64(i_28496);
                int64_t new_index_28500 = i_28349 + binop_x_28499;
                float x_27722 = ((float *) mem_28760)[new_index_28500];
                float res_27721 = x_27722 + redout_28348;
                float redout_tmp_28842 = res_27721;
                
                redout_28348 = redout_tmp_28842;
            }
            res_27718 = redout_28348;
            
            float res_27723 = fmax32(0.0F, res_27718);
            float res_27716 = res_27723 + redout_28350;
            float redout_tmp_28841 = res_27716;
            
            redout_28350 = redout_tmp_28841;
        }
        res_27713 = redout_28350;
        
        float res_27724 = res_27713 / res_27476;
        float negate_arg_27725 = 1.0e-2F * res_27485;
        float exp_arg_27726 = 0.0F - negate_arg_27725;
        float res_27727 = fpow32(2.7182817F, exp_arg_27726);
        float x_27728 = 1.0F - res_27727;
        float B_27729 = x_27728 / 1.0e-2F;
        float x_27730 = B_27729 - res_27485;
        float x_27731 = 4.4999997e-6F * x_27730;
        float A1_27732 = x_27731 / 1.0e-4F;
        float y_27733 = fpow32(B_27729, 2.0F);
        float x_27734 = 1.0000001e-6F * y_27733;
        float A2_27735 = x_27734 / 4.0e-2F;
        float exp_arg_27736 = A1_27732 - A2_27735;
        float res_27737 = fpow32(2.7182817F, exp_arg_27736);
        float negate_arg_27738 = 5.0e-2F * B_27729;
        float exp_arg_27739 = 0.0F - negate_arg_27738;
        float res_27740 = fpow32(2.7182817F, exp_arg_27739);
        float res_27741 = res_27737 * res_27740;
        float res_27742 = res_27724 * res_27741;
        float res_27481 = res_27742 + redout_28352;
        float redout_tmp_28822 = res_27481;
        
        redout_28352 = redout_tmp_28822;
    }
    res_27478 = redout_28352;
    
    float CVA_27743 = 6.0e-3F * res_27478;
    
    scalar_out_28808 = CVA_27743;
    *out_scalar_out_28868 = scalar_out_28808;
    
  cleanup:
    { }
    free(mem_28511);
    free(mem_28513);
    free(mem_28515);
    free(mem_28517);
    free(mem_28519);
    free(mem_28521);
    free(mem_28523);
    free(mem_28582);
    free(mem_28594);
    free(mem_28608);
    free(mem_28637);
    free(mem_28652);
    free(mem_28662);
    free(mem_28664);
    free(mem_28690);
    free(mem_28704);
    free(mem_28718);
    free(mem_28720);
    free(mem_28746);
    free(mem_28760);
    return err;
}
static int futrts_test2(struct futhark_context *ctx,
                        float *out_scalar_out_28892, int64_t paths_27744,
                        int64_t steps_27745, int64_t numswaps_27746)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_28511_cached_sizze_28893 = 0;
    char *mem_28511 = NULL;
    size_t mem_28513_cached_sizze_28894 = 0;
    char *mem_28513 = NULL;
    size_t mem_28515_cached_sizze_28895 = 0;
    char *mem_28515 = NULL;
    size_t mem_28553_cached_sizze_28896 = 0;
    char *mem_28553 = NULL;
    size_t mem_28555_cached_sizze_28897 = 0;
    char *mem_28555 = NULL;
    size_t mem_28557_cached_sizze_28898 = 0;
    char *mem_28557 = NULL;
    size_t mem_28559_cached_sizze_28899 = 0;
    char *mem_28559 = NULL;
    size_t mem_28610_cached_sizze_28900 = 0;
    char *mem_28610 = NULL;
    size_t mem_28622_cached_sizze_28901 = 0;
    char *mem_28622 = NULL;
    size_t mem_28636_cached_sizze_28902 = 0;
    char *mem_28636 = NULL;
    size_t mem_28665_cached_sizze_28903 = 0;
    char *mem_28665 = NULL;
    size_t mem_28677_cached_sizze_28904 = 0;
    char *mem_28677 = NULL;
    size_t mem_28687_cached_sizze_28905 = 0;
    char *mem_28687 = NULL;
    size_t mem_28689_cached_sizze_28906 = 0;
    char *mem_28689 = NULL;
    size_t mem_28715_cached_sizze_28907 = 0;
    char *mem_28715 = NULL;
    size_t mem_28729_cached_sizze_28908 = 0;
    char *mem_28729 = NULL;
    size_t mem_28743_cached_sizze_28909 = 0;
    char *mem_28743 = NULL;
    size_t mem_28745_cached_sizze_28910 = 0;
    char *mem_28745 = NULL;
    size_t mem_28771_cached_sizze_28911 = 0;
    char *mem_28771 = NULL;
    size_t mem_28785_cached_sizze_28912 = 0;
    char *mem_28785 = NULL;
    float scalar_out_28808;
    bool bounds_invalid_upwards_27747 = slt64(numswaps_27746, 0);
    bool valid_27748 = !bounds_invalid_upwards_27747;
    bool range_valid_c_27749;
    
    if (!valid_27748) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", numswaps_27746,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:172:29-62\n   #4  cva.fut:172:19-93\n   #5  cva.fut:170:1-176:75\n");
        return 1;
    }
    
    float res_27763 = sitofp_i64_f32(numswaps_27746);
    float res_27764 = res_27763 - 1.0F;
    int64_t bytes_28510 = 4 * numswaps_27746;
    
    if (mem_28511_cached_sizze_28893 < (size_t) bytes_28510) {
        mem_28511 = realloc(mem_28511, bytes_28510);
        mem_28511_cached_sizze_28893 = bytes_28510;
    }
    
    int64_t bytes_28512 = 8 * numswaps_27746;
    
    if (mem_28513_cached_sizze_28894 < (size_t) bytes_28512) {
        mem_28513 = realloc(mem_28513, bytes_28512);
        mem_28513_cached_sizze_28894 = bytes_28512;
    }
    if (mem_28515_cached_sizze_28895 < (size_t) bytes_28510) {
        mem_28515 = realloc(mem_28515, bytes_28510);
        mem_28515_cached_sizze_28895 = bytes_28510;
    }
    
    float res_27776;
    float redout_28230 = -INFINITY;
    
    for (int64_t i_28234 = 0; i_28234 < numswaps_27746; i_28234++) {
        int32_t res_27784 = sext_i64_i32(i_28234);
        int32_t x_27785 = lshr32(res_27784, 16);
        int32_t x_27786 = res_27784 ^ x_27785;
        int32_t x_27787 = mul32(73244475, x_27786);
        int32_t x_27788 = lshr32(x_27787, 16);
        int32_t x_27789 = x_27787 ^ x_27788;
        int32_t x_27790 = mul32(73244475, x_27789);
        int32_t x_27791 = lshr32(x_27790, 16);
        int32_t x_27792 = x_27790 ^ x_27791;
        int32_t unsign_arg_27793 = 281253711 ^ x_27792;
        int32_t unsign_arg_27794 = mul32(48271, unsign_arg_27793);
        int32_t unsign_arg_27795 = umod32(unsign_arg_27794, 2147483647);
        float res_27796 = uitofp_i32_f32(unsign_arg_27795);
        float res_27797 = res_27796 / 2.1474836e9F;
        float res_27798 = 2.0F * res_27797;
        float res_27799 = res_27764 * res_27797;
        float res_27800 = 1.0F + res_27799;
        int64_t res_27801 = fptosi_f32_i64(res_27800);
        float res_27807 = -1.0F + res_27798;
        float res_27808 = sitofp_i64_f32(res_27801);
        float res_27809 = res_27798 * res_27808;
        float res_27782 = fmax32(res_27809, redout_28230);
        
        ((float *) mem_28511)[i_28234] = res_27807;
        ((int64_t *) mem_28513)[i_28234] = res_27801;
        ((float *) mem_28515)[i_28234] = res_27798;
        
        float redout_tmp_28809 = res_27782;
        
        redout_28230 = redout_tmp_28809;
    }
    res_27776 = redout_28230;
    
    float res_27814 = sitofp_i64_f32(steps_27745);
    float dt_27815 = res_27776 / res_27814;
    
    if (mem_28553_cached_sizze_28896 < (size_t) bytes_28510) {
        mem_28553 = realloc(mem_28553, bytes_28510);
        mem_28553_cached_sizze_28896 = bytes_28510;
    }
    if (mem_28555_cached_sizze_28897 < (size_t) bytes_28510) {
        mem_28555 = realloc(mem_28555, bytes_28510);
        mem_28555_cached_sizze_28897 = bytes_28510;
    }
    if (mem_28557_cached_sizze_28898 < (size_t) bytes_28512) {
        mem_28557 = realloc(mem_28557, bytes_28512);
        mem_28557_cached_sizze_28898 = bytes_28512;
    }
    if (mem_28559_cached_sizze_28899 < (size_t) bytes_28510) {
        mem_28559 = realloc(mem_28559, bytes_28510);
        mem_28559_cached_sizze_28899 = bytes_28510;
    }
    for (int64_t i_28248 = 0; i_28248 < numswaps_27746; i_28248++) {
        float res_27825 = ((float *) mem_28515)[i_28248];
        int64_t res_27826 = ((int64_t *) mem_28513)[i_28248];
        bool bounds_invalid_upwards_27828 = slt64(res_27826, 1);
        bool valid_27829 = !bounds_invalid_upwards_27828;
        bool range_valid_c_27830;
        
        if (!valid_27829) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_27826,
                          " is invalid.",
                          "-> #0  cva.fut:55:29-48\n   #1  cva.fut:96:25-65\n   #2  cva.fut:111:16-62\n   #3  cva.fut:107:17-111:85\n   #4  cva.fut:176:8-67\n   #5  cva.fut:170:1-176:75\n");
            return 1;
        }
        
        bool y_27832 = slt64(0, res_27826);
        bool index_certs_27833;
        
        if (!y_27832) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_27826, "].",
                                   "-> #0  cva.fut:97:47-70\n   #1  cva.fut:111:16-62\n   #2  cva.fut:107:17-111:85\n   #3  cva.fut:176:8-67\n   #4  cva.fut:170:1-176:75\n");
            return 1;
        }
        
        float binop_y_27834 = sitofp_i64_f32(res_27826);
        float index_primexp_27835 = res_27825 * binop_y_27834;
        float negate_arg_27836 = 1.0e-2F * index_primexp_27835;
        float exp_arg_27837 = 0.0F - negate_arg_27836;
        float res_27838 = fpow32(2.7182817F, exp_arg_27837);
        float x_27839 = 1.0F - res_27838;
        float B_27840 = x_27839 / 1.0e-2F;
        float x_27841 = B_27840 - index_primexp_27835;
        float x_27842 = 4.4999997e-6F * x_27841;
        float A1_27843 = x_27842 / 1.0e-4F;
        float y_27844 = fpow32(B_27840, 2.0F);
        float x_27845 = 1.0000001e-6F * y_27844;
        float A2_27846 = x_27845 / 4.0e-2F;
        float exp_arg_27847 = A1_27843 - A2_27846;
        float res_27848 = fpow32(2.7182817F, exp_arg_27847);
        float negate_arg_27849 = 5.0e-2F * B_27840;
        float exp_arg_27850 = 0.0F - negate_arg_27849;
        float res_27851 = fpow32(2.7182817F, exp_arg_27850);
        float res_27852 = res_27848 * res_27851;
        float res_27853;
        float redout_28238 = 0.0F;
        
        for (int64_t i_28239 = 0; i_28239 < res_27826; i_28239++) {
            int64_t index_primexp_28366 = add64(1, i_28239);
            float res_27858 = sitofp_i64_f32(index_primexp_28366);
            float res_27859 = res_27825 * res_27858;
            float negate_arg_27860 = 1.0e-2F * res_27859;
            float exp_arg_27861 = 0.0F - negate_arg_27860;
            float res_27862 = fpow32(2.7182817F, exp_arg_27861);
            float x_27863 = 1.0F - res_27862;
            float B_27864 = x_27863 / 1.0e-2F;
            float x_27865 = B_27864 - res_27859;
            float x_27866 = 4.4999997e-6F * x_27865;
            float A1_27867 = x_27866 / 1.0e-4F;
            float y_27868 = fpow32(B_27864, 2.0F);
            float x_27869 = 1.0000001e-6F * y_27868;
            float A2_27870 = x_27869 / 4.0e-2F;
            float exp_arg_27871 = A1_27867 - A2_27870;
            float res_27872 = fpow32(2.7182817F, exp_arg_27871);
            float negate_arg_27873 = 5.0e-2F * B_27864;
            float exp_arg_27874 = 0.0F - negate_arg_27873;
            float res_27875 = fpow32(2.7182817F, exp_arg_27874);
            float res_27876 = res_27872 * res_27875;
            float res_27856 = res_27876 + redout_28238;
            float redout_tmp_28817 = res_27856;
            
            redout_28238 = redout_tmp_28817;
        }
        res_27853 = redout_28238;
        
        float x_27877 = 1.0F - res_27852;
        float y_27878 = res_27825 * res_27853;
        float res_27879 = x_27877 / y_27878;
        
        ((float *) mem_28553)[i_28248] = res_27879;
        memmove(mem_28555 + i_28248 * 4, mem_28511 + i_28248 * 4,
                (int32_t) sizeof(float));
        memmove(mem_28557 + i_28248 * 8, mem_28513 + i_28248 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_28559 + i_28248 * 4, mem_28515 + i_28248 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_27880 = res_27814 / res_27776;
    bool bounds_invalid_upwards_27881 = slt64(steps_27745, 1);
    bool valid_27882 = !bounds_invalid_upwards_27881;
    bool range_valid_c_27883;
    
    if (!valid_27882) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_27745,
                               " is invalid.",
                               "-> #0  cva.fut:61:56-67\n   #1  cva.fut:113:17-44\n   #2  cva.fut:176:8-67\n   #3  cva.fut:170:1-176:75\n");
        return 1;
    }
    
    bool bounds_invalid_upwards_27885 = slt64(paths_27744, 0);
    bool valid_27886 = !bounds_invalid_upwards_27885;
    bool range_valid_c_27887;
    
    if (!valid_27886) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_27744,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:116:19-49\n   #4  cva.fut:176:8-67\n   #5  cva.fut:170:1-176:75\n");
        return 1;
    }
    
    int64_t upper_bound_27890 = sub64(steps_27745, 1);
    float res_27891;
    
    res_27891 = futrts_sqrt32(dt_27815);
    
    int64_t binop_x_28609 = paths_27744 * steps_27745;
    int64_t bytes_28608 = 4 * binop_x_28609;
    
    if (mem_28610_cached_sizze_28900 < (size_t) bytes_28608) {
        mem_28610 = realloc(mem_28610, bytes_28608);
        mem_28610_cached_sizze_28900 = bytes_28608;
    }
    
    int64_t bytes_28621 = 4 * steps_27745;
    
    if (mem_28622_cached_sizze_28901 < (size_t) bytes_28621) {
        mem_28622 = realloc(mem_28622, bytes_28621);
        mem_28622_cached_sizze_28901 = bytes_28621;
    }
    if (mem_28636_cached_sizze_28902 < (size_t) bytes_28621) {
        mem_28636 = realloc(mem_28636, bytes_28621);
        mem_28636_cached_sizze_28902 = bytes_28621;
    }
    for (int64_t i_28818 = 0; i_28818 < steps_27745; i_28818++) {
        ((float *) mem_28636)[i_28818] = 5.0e-2F;
    }
    for (int64_t i_28259 = 0; i_28259 < paths_27744; i_28259++) {
        int32_t res_27894 = sext_i64_i32(i_28259);
        int32_t x_27895 = lshr32(res_27894, 16);
        int32_t x_27896 = res_27894 ^ x_27895;
        int32_t x_27897 = mul32(73244475, x_27896);
        int32_t x_27898 = lshr32(x_27897, 16);
        int32_t x_27899 = x_27897 ^ x_27898;
        int32_t x_27900 = mul32(73244475, x_27899);
        int32_t x_27901 = lshr32(x_27900, 16);
        int32_t x_27902 = x_27900 ^ x_27901;
        int32_t unsign_arg_27903 = 777822902 ^ x_27902;
        int32_t unsign_arg_27904 = mul32(48271, unsign_arg_27903);
        int32_t unsign_arg_27905 = umod32(unsign_arg_27904, 2147483647);
        
        for (int64_t i_28255 = 0; i_28255 < steps_27745; i_28255++) {
            int32_t res_27908 = sext_i64_i32(i_28255);
            int32_t x_27909 = lshr32(res_27908, 16);
            int32_t x_27910 = res_27908 ^ x_27909;
            int32_t x_27911 = mul32(73244475, x_27910);
            int32_t x_27912 = lshr32(x_27911, 16);
            int32_t x_27913 = x_27911 ^ x_27912;
            int32_t x_27914 = mul32(73244475, x_27913);
            int32_t x_27915 = lshr32(x_27914, 16);
            int32_t x_27916 = x_27914 ^ x_27915;
            int32_t unsign_arg_27917 = unsign_arg_27905 ^ x_27916;
            int32_t unsign_arg_27918 = mul32(48271, unsign_arg_27917);
            int32_t unsign_arg_27919 = umod32(unsign_arg_27918, 2147483647);
            int32_t unsign_arg_27920 = mul32(48271, unsign_arg_27919);
            int32_t unsign_arg_27921 = umod32(unsign_arg_27920, 2147483647);
            float res_27922 = uitofp_i32_f32(unsign_arg_27919);
            float res_27923 = res_27922 / 2.1474836e9F;
            float res_27924 = uitofp_i32_f32(unsign_arg_27921);
            float res_27925 = res_27924 / 2.1474836e9F;
            float res_27926;
            
            res_27926 = futrts_log32(res_27923);
            
            float res_27927 = -2.0F * res_27926;
            float res_27928;
            
            res_27928 = futrts_sqrt32(res_27927);
            
            float res_27929 = 6.2831855F * res_27925;
            float res_27930;
            
            res_27930 = futrts_cos32(res_27929);
            
            float res_27931 = res_27928 * res_27930;
            
            ((float *) mem_28622)[i_28255] = res_27931;
        }
        memmove(mem_28610 + i_28259 * steps_27745 * 4, mem_28636 + 0,
                steps_27745 * (int32_t) sizeof(float));
        for (int64_t i_27934 = 0; i_27934 < upper_bound_27890; i_27934++) {
            bool y_27936 = slt64(i_27934, steps_27745);
            bool index_certs_27937;
            
            if (!y_27936) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27934,
                              "] out of bounds for array of shape [",
                              steps_27745, "].",
                              "-> #0  cva.fut:72:97-104\n   #1  cva.fut:124:32-62\n   #2  cva.fut:124:22-69\n   #3  cva.fut:176:8-67\n   #4  cva.fut:170:1-176:75\n");
                return 1;
            }
            
            float shortstep_arg_27938 = ((float *) mem_28622)[i_27934];
            float shortstep_arg_27939 = ((float *) mem_28610)[i_28259 *
                                                              steps_27745 +
                                                              i_27934];
            float y_27940 = 5.0e-2F - shortstep_arg_27939;
            float x_27941 = 1.0e-2F * y_27940;
            float x_27942 = dt_27815 * x_27941;
            float x_27943 = res_27891 * shortstep_arg_27938;
            float y_27944 = 1.0e-3F * x_27943;
            float delta_r_27945 = x_27942 + y_27944;
            float res_27946 = shortstep_arg_27939 + delta_r_27945;
            int64_t i_27947 = add64(1, i_27934);
            bool x_27948 = sle64(0, i_27947);
            bool y_27949 = slt64(i_27947, steps_27745);
            bool bounds_check_27950 = x_27948 && y_27949;
            bool index_certs_27951;
            
            if (!bounds_check_27950) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27947,
                              "] out of bounds for array of shape [",
                              steps_27745, "].",
                              "-> #0  cva.fut:72:58-105\n   #1  cva.fut:124:32-62\n   #2  cva.fut:124:22-69\n   #3  cva.fut:176:8-67\n   #4  cva.fut:170:1-176:75\n");
                return 1;
            }
            ((float *) mem_28610)[i_28259 * steps_27745 + i_27947] = res_27946;
        }
    }
    
    int64_t flat_dim_27955 = paths_27744 * numswaps_27746;
    float res_27959 = sitofp_i64_f32(paths_27744);
    int64_t bytes_28663 = 4 * flat_dim_27955;
    
    if (mem_28665_cached_sizze_28903 < (size_t) bytes_28663) {
        mem_28665 = realloc(mem_28665, bytes_28663);
        mem_28665_cached_sizze_28903 = bytes_28663;
    }
    if (mem_28677_cached_sizze_28904 < (size_t) bytes_28510) {
        mem_28677 = realloc(mem_28677, bytes_28510);
        mem_28677_cached_sizze_28904 = bytes_28510;
    }
    
    int64_t bytes_28686 = 8 * flat_dim_27955;
    
    if (mem_28687_cached_sizze_28905 < (size_t) bytes_28686) {
        mem_28687 = realloc(mem_28687, bytes_28686);
        mem_28687_cached_sizze_28905 = bytes_28686;
    }
    if (mem_28689_cached_sizze_28906 < (size_t) bytes_28686) {
        mem_28689 = realloc(mem_28689, bytes_28686);
        mem_28689_cached_sizze_28906 = bytes_28686;
    }
    
    float res_27961;
    float redout_28361 = 0.0F;
    
    for (int64_t i_28362 = 0; i_28362 < steps_27745; i_28362++) {
        int64_t index_primexp_28497 = add64(1, i_28362);
        float res_27967 = sitofp_i64_f32(index_primexp_28497);
        float res_27968 = res_27967 / sims_per_year_27880;
        
        for (int64_t i_28263 = 0; i_28263 < paths_27744; i_28263++) {
            float x_27970 = ((float *) mem_28610)[i_28263 * steps_27745 +
                                                  i_28362];
            
            for (int64_t i_28824 = 0; i_28824 < numswaps_27746; i_28824++) {
                ((float *) mem_28677)[i_28824] = x_27970;
            }
            memmove(mem_28665 + i_28263 * numswaps_27746 * 4, mem_28677 + 0,
                    numswaps_27746 * (int32_t) sizeof(float));
        }
        
        int64_t discard_28273;
        int64_t scanacc_28267 = 0;
        
        for (int64_t i_28270 = 0; i_28270 < flat_dim_27955; i_28270++) {
            int64_t binop_x_28377 = squot64(i_28270, numswaps_27746);
            int64_t binop_y_28378 = numswaps_27746 * binop_x_28377;
            int64_t new_index_28379 = i_28270 - binop_y_28378;
            int64_t x_27977 = ((int64_t *) mem_28557)[new_index_28379];
            float x_27978 = ((float *) mem_28559)[new_index_28379];
            float x_27979 = res_27968 / x_27978;
            float ceil_arg_27980 = x_27979 - 1.0F;
            float res_27981;
            
            res_27981 = futrts_ceil32(ceil_arg_27980);
            
            int64_t res_27982 = fptosi_f32_i64(res_27981);
            int64_t max_arg_27983 = sub64(x_27977, res_27982);
            int64_t res_27984 = smax64(0, max_arg_27983);
            bool cond_27985 = res_27984 == 0;
            int64_t res_27986;
            
            if (cond_27985) {
                res_27986 = 1;
            } else {
                res_27986 = res_27984;
            }
            
            int64_t res_27976 = add64(res_27986, scanacc_28267);
            
            ((int64_t *) mem_28687)[i_28270] = res_27976;
            ((int64_t *) mem_28689)[i_28270] = res_27986;
            
            int64_t scanacc_tmp_28825 = res_27976;
            
            scanacc_28267 = scanacc_tmp_28825;
        }
        discard_28273 = scanacc_28267;
        
        int64_t res_27988;
        int64_t redout_28274 = 0;
        
        for (int64_t i_28275 = 0; i_28275 < flat_dim_27955; i_28275++) {
            int64_t x_27992 = ((int64_t *) mem_28689)[i_28275];
            int64_t res_27991 = add64(x_27992, redout_28274);
            int64_t redout_tmp_28828 = res_27991;
            
            redout_28274 = redout_tmp_28828;
        }
        res_27988 = redout_28274;
        
        bool bounds_invalid_upwards_27993 = slt64(res_27988, 0);
        bool valid_27994 = !bounds_invalid_upwards_27993;
        bool range_valid_c_27995;
        
        if (!valid_27994) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", res_27988,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:176:8-67\n   #10 cva.fut:170:1-176:75\n");
            return 1;
        }
        
        int64_t bytes_28714 = 8 * res_27988;
        
        if (mem_28715_cached_sizze_28907 < (size_t) bytes_28714) {
            mem_28715 = realloc(mem_28715, bytes_28714);
            mem_28715_cached_sizze_28907 = bytes_28714;
        }
        for (int64_t i_28829 = 0; i_28829 < res_27988; i_28829++) {
            ((int64_t *) mem_28715)[i_28829] = 0;
        }
        for (int64_t iter_28276 = 0; iter_28276 < flat_dim_27955;
             iter_28276++) {
            int64_t i_p_o_28384 = add64(-1, iter_28276);
            int64_t rot_i_28385 = smod64(i_p_o_28384, flat_dim_27955);
            int64_t pixel_28279 = ((int64_t *) mem_28687)[rot_i_28385];
            bool cond_28003 = iter_28276 == 0;
            int64_t res_28004;
            
            if (cond_28003) {
                res_28004 = 0;
            } else {
                res_28004 = pixel_28279;
            }
            
            bool less_than_zzero_28280 = slt64(res_28004, 0);
            bool greater_than_sizze_28281 = sle64(res_27988, res_28004);
            bool outside_bounds_dim_28282 = less_than_zzero_28280 ||
                 greater_than_sizze_28281;
            
            if (!outside_bounds_dim_28282) {
                int64_t read_hist_28284 = ((int64_t *) mem_28715)[res_28004];
                int64_t res_28000 = smax64(iter_28276, read_hist_28284);
                
                ((int64_t *) mem_28715)[res_28004] = res_28000;
            }
        }
        if (mem_28729_cached_sizze_28908 < (size_t) bytes_28714) {
            mem_28729 = realloc(mem_28729, bytes_28714);
            mem_28729_cached_sizze_28908 = bytes_28714;
        }
        
        int64_t discard_28297;
        int64_t scanacc_28290 = 0;
        
        for (int64_t i_28293 = 0; i_28293 < res_27988; i_28293++) {
            int64_t x_28014 = ((int64_t *) mem_28715)[i_28293];
            bool res_28015 = slt64(0, x_28014);
            int64_t res_28012;
            
            if (res_28015) {
                res_28012 = x_28014;
            } else {
                int64_t res_28013 = add64(x_28014, scanacc_28290);
                
                res_28012 = res_28013;
            }
            ((int64_t *) mem_28729)[i_28293] = res_28012;
            
            int64_t scanacc_tmp_28831 = res_28012;
            
            scanacc_28290 = scanacc_tmp_28831;
        }
        discard_28297 = scanacc_28290;
        
        int64_t bytes_28742 = 4 * res_27988;
        
        if (mem_28743_cached_sizze_28909 < (size_t) bytes_28742) {
            mem_28743 = realloc(mem_28743, bytes_28742);
            mem_28743_cached_sizze_28909 = bytes_28742;
        }
        if (mem_28745_cached_sizze_28910 < (size_t) res_27988) {
            mem_28745 = realloc(mem_28745, res_27988);
            mem_28745_cached_sizze_28910 = res_27988;
        }
        
        int64_t inpacc_28024;
        float inpacc_28026;
        int64_t inpacc_28032;
        float inpacc_28034;
        
        inpacc_28032 = 0;
        inpacc_28034 = 0.0F;
        for (int64_t i_28336 = 0; i_28336 < res_27988; i_28336++) {
            int64_t x_28399 = ((int64_t *) mem_28729)[i_28336];
            int64_t i_p_o_28401 = add64(-1, i_28336);
            int64_t rot_i_28402 = smod64(i_p_o_28401, res_27988);
            int64_t x_28403 = ((int64_t *) mem_28729)[rot_i_28402];
            bool res_28404 = x_28399 == x_28403;
            bool res_28405 = !res_28404;
            int64_t res_28068;
            
            if (res_28405) {
                res_28068 = 1;
            } else {
                int64_t res_28069 = add64(1, inpacc_28032);
                
                res_28068 = res_28069;
            }
            
            int64_t res_28083;
            
            if (res_28405) {
                res_28083 = 1;
            } else {
                int64_t res_28084 = add64(1, inpacc_28032);
                
                res_28083 = res_28084;
            }
            
            int64_t res_28085 = sub64(res_28083, 1);
            bool x_28420 = sle64(0, x_28399);
            bool y_28421 = slt64(x_28399, flat_dim_27955);
            bool bounds_check_28422 = x_28420 && y_28421;
            bool index_certs_28423;
            
            if (!bounds_check_28422) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", x_28399,
                              "] out of bounds for array of shape [",
                              flat_dim_27955, "].",
                              "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:176:8-67\n   #10 cva.fut:170:1-176:75\n");
                return 1;
            }
            
            int64_t new_index_28424 = squot64(x_28399, numswaps_27746);
            int64_t binop_y_28425 = numswaps_27746 * new_index_28424;
            int64_t new_index_28426 = x_28399 - binop_y_28425;
            float lifted_0_get_arg_28427 =
                  ((float *) mem_28665)[new_index_28424 * numswaps_27746 +
                                        new_index_28426];
            float lifted_0_get_arg_28428 =
                  ((float *) mem_28553)[new_index_28426];
            float lifted_0_get_arg_28429 =
                  ((float *) mem_28555)[new_index_28426];
            int64_t lifted_0_get_arg_28430 =
                    ((int64_t *) mem_28557)[new_index_28426];
            float lifted_0_get_arg_28431 =
                  ((float *) mem_28559)[new_index_28426];
            float x_28432 = res_27968 / lifted_0_get_arg_28431;
            float ceil_arg_28433 = x_28432 - 1.0F;
            float res_28434;
            
            res_28434 = futrts_ceil32(ceil_arg_28433);
            
            int64_t res_28435 = fptosi_f32_i64(res_28434);
            int64_t max_arg_28436 = sub64(lifted_0_get_arg_28430, res_28435);
            int64_t res_28437 = smax64(0, max_arg_28436);
            bool cond_28438 = res_28437 == 0;
            float res_28439;
            
            if (cond_28438) {
                res_28439 = 0.0F;
            } else {
                float res_28440;
                
                res_28440 = futrts_ceil32(x_28432);
                
                float start_28441 = lifted_0_get_arg_28431 * res_28440;
                float res_28442;
                
                res_28442 = futrts_ceil32(ceil_arg_28433);
                
                int64_t res_28443 = fptosi_f32_i64(res_28442);
                int64_t max_arg_28444 = sub64(lifted_0_get_arg_28430,
                                              res_28443);
                int64_t res_28445 = smax64(0, max_arg_28444);
                int64_t sizze_28446 = sub64(res_28445, 1);
                bool cond_28447 = res_28085 == 0;
                float res_28448;
                
                if (cond_28447) {
                    res_28448 = 1.0F;
                } else {
                    res_28448 = 0.0F;
                }
                
                bool cond_28449 = slt64(0, res_28085);
                float res_28450;
                
                if (cond_28449) {
                    float y_28451 = lifted_0_get_arg_28428 *
                          lifted_0_get_arg_28431;
                    float res_28452 = res_28448 - y_28451;
                    
                    res_28450 = res_28452;
                } else {
                    res_28450 = res_28448;
                }
                
                bool cond_28453 = res_28085 == sizze_28446;
                float res_28454;
                
                if (cond_28453) {
                    float res_28455 = res_28450 - 1.0F;
                    
                    res_28454 = res_28455;
                } else {
                    res_28454 = res_28450;
                }
                
                float res_28456 = lifted_0_get_arg_28429 * res_28454;
                float res_28457 = sitofp_i64_f32(res_28085);
                float y_28458 = lifted_0_get_arg_28431 * res_28457;
                float bondprice_arg_28459 = start_28441 + y_28458;
                float y_28460 = bondprice_arg_28459 - res_27968;
                float negate_arg_28461 = 1.0e-2F * y_28460;
                float exp_arg_28462 = 0.0F - negate_arg_28461;
                float res_28463 = fpow32(2.7182817F, exp_arg_28462);
                float x_28464 = 1.0F - res_28463;
                float B_28465 = x_28464 / 1.0e-2F;
                float x_28466 = B_28465 - bondprice_arg_28459;
                float x_28467 = res_27968 + x_28466;
                float x_28468 = 4.4999997e-6F * x_28467;
                float A1_28469 = x_28468 / 1.0e-4F;
                float y_28470 = fpow32(B_28465, 2.0F);
                float x_28471 = 1.0000001e-6F * y_28470;
                float A2_28472 = x_28471 / 4.0e-2F;
                float exp_arg_28473 = A1_28469 - A2_28472;
                float res_28474 = fpow32(2.7182817F, exp_arg_28473);
                float negate_arg_28475 = lifted_0_get_arg_28427 * B_28465;
                float exp_arg_28476 = 0.0F - negate_arg_28475;
                float res_28477 = fpow32(2.7182817F, exp_arg_28476);
                float res_28478 = res_28474 * res_28477;
                float x_28479 = res_28456 * res_28478;
                float res_28480 = lifted_0_get_arg_28429 * x_28479;
                
                res_28439 = res_28480;
            }
            
            float res_28156;
            
            if (res_28405) {
                res_28156 = res_28439;
            } else {
                float res_28157 = inpacc_28034 + res_28439;
                
                res_28156 = res_28157;
            }
            
            float res_28159;
            
            if (res_28405) {
                res_28159 = res_28439;
            } else {
                float res_28160 = inpacc_28034 + res_28439;
                
                res_28159 = res_28160;
            }
            ((float *) mem_28743)[i_28336] = res_28156;
            ((bool *) mem_28745)[i_28336] = res_28405;
            
            int64_t inpacc_tmp_28833 = res_28068;
            float inpacc_tmp_28834 = res_28159;
            
            inpacc_28032 = inpacc_tmp_28833;
            inpacc_28034 = inpacc_tmp_28834;
        }
        inpacc_28024 = inpacc_28032;
        inpacc_28026 = inpacc_28034;
        if (mem_28771_cached_sizze_28911 < (size_t) bytes_28714) {
            mem_28771 = realloc(mem_28771, bytes_28714);
            mem_28771_cached_sizze_28911 = bytes_28714;
        }
        
        int64_t discard_28345;
        int64_t scanacc_28341 = 0;
        
        for (int64_t i_28343 = 0; i_28343 < res_27988; i_28343++) {
            int64_t i_p_o_28491 = add64(1, i_28343);
            int64_t rot_i_28492 = smod64(i_p_o_28491, res_27988);
            bool x_28173 = ((bool *) mem_28745)[rot_i_28492];
            int64_t res_28174 = btoi_bool_i64(x_28173);
            int64_t res_28172 = add64(res_28174, scanacc_28341);
            
            ((int64_t *) mem_28771)[i_28343] = res_28172;
            
            int64_t scanacc_tmp_28837 = res_28172;
            
            scanacc_28341 = scanacc_tmp_28837;
        }
        discard_28345 = scanacc_28341;
        
        bool cond_28175 = slt64(0, res_27988);
        int64_t num_segments_28176;
        
        if (cond_28175) {
            int64_t i_28177 = sub64(res_27988, 1);
            bool x_28178 = sle64(0, i_28177);
            bool y_28179 = slt64(i_28177, res_27988);
            bool bounds_check_28180 = x_28178 && y_28179;
            bool index_certs_28181;
            
            if (!bounds_check_28180) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_28177,
                              "] out of bounds for array of shape [", res_27988,
                              "].",
                              "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva.fut:134:38-97\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:127:44-137:50\n   #8  cva.fut:176:8-67\n   #9  cva.fut:170:1-176:75\n");
                return 1;
            }
            
            int64_t res_28182 = ((int64_t *) mem_28771)[i_28177];
            
            num_segments_28176 = res_28182;
        } else {
            num_segments_28176 = 0;
        }
        
        bool bounds_invalid_upwards_28183 = slt64(num_segments_28176, 0);
        bool valid_28184 = !bounds_invalid_upwards_28183;
        bool range_valid_c_28185;
        
        if (!valid_28184) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", num_segments_28176,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:134:38-97\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:127:44-137:50\n   #9  cva.fut:176:8-67\n   #10 cva.fut:170:1-176:75\n");
            return 1;
        }
        
        int64_t bytes_28784 = 4 * num_segments_28176;
        
        if (mem_28785_cached_sizze_28912 < (size_t) bytes_28784) {
            mem_28785 = realloc(mem_28785, bytes_28784);
            mem_28785_cached_sizze_28912 = bytes_28784;
        }
        for (int64_t i_28839 = 0; i_28839 < num_segments_28176; i_28839++) {
            ((float *) mem_28785)[i_28839] = 0.0F;
        }
        for (int64_t write_iter_28346 = 0; write_iter_28346 < res_27988;
             write_iter_28346++) {
            int64_t write_iv_28348 = ((int64_t *) mem_28771)[write_iter_28346];
            int64_t i_p_o_28494 = add64(1, write_iter_28346);
            int64_t rot_i_28495 = smod64(i_p_o_28494, res_27988);
            bool write_iv_28349 = ((bool *) mem_28745)[rot_i_28495];
            int64_t res_28191;
            
            if (write_iv_28349) {
                int64_t res_28192 = sub64(write_iv_28348, 1);
                
                res_28191 = res_28192;
            } else {
                res_28191 = -1;
            }
            
            bool less_than_zzero_28351 = slt64(res_28191, 0);
            bool greater_than_sizze_28352 = sle64(num_segments_28176,
                                                  res_28191);
            bool outside_bounds_dim_28353 = less_than_zzero_28351 ||
                 greater_than_sizze_28352;
            
            if (!outside_bounds_dim_28353) {
                memmove(mem_28785 + res_28191 * 4, mem_28743 +
                        write_iter_28346 * 4, (int32_t) sizeof(float));
            }
        }
        
        bool dim_match_28193 = flat_dim_27955 == num_segments_28176;
        bool empty_or_match_cert_28194;
        
        if (!dim_match_28193) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Value of (core language) shape (",
                                   num_segments_28176,
                                   ") cannot match shape of type `[",
                                   flat_dim_27955, "]b`.",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva.fut:134:38-97\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:127:44-137:50\n   #5  cva.fut:176:8-67\n   #6  cva.fut:170:1-176:75\n");
            return 1;
        }
        
        float res_28196;
        float redout_28359 = 0.0F;
        
        for (int64_t i_28360 = 0; i_28360 < paths_27744; i_28360++) {
            int64_t binop_x_28498 = numswaps_27746 * i_28360;
            float res_28201;
            float redout_28357 = 0.0F;
            
            for (int64_t i_28358 = 0; i_28358 < numswaps_27746; i_28358++) {
                int64_t new_index_28499 = i_28358 + binop_x_28498;
                float x_28205 = ((float *) mem_28785)[new_index_28499];
                float res_28204 = x_28205 + redout_28357;
                float redout_tmp_28842 = res_28204;
                
                redout_28357 = redout_tmp_28842;
            }
            res_28201 = redout_28357;
            
            float res_28206 = fmax32(0.0F, res_28201);
            float res_28199 = res_28206 + redout_28359;
            float redout_tmp_28841 = res_28199;
            
            redout_28359 = redout_tmp_28841;
        }
        res_28196 = redout_28359;
        
        float res_28207 = res_28196 / res_27959;
        float negate_arg_28208 = 1.0e-2F * res_27968;
        float exp_arg_28209 = 0.0F - negate_arg_28208;
        float res_28210 = fpow32(2.7182817F, exp_arg_28209);
        float x_28211 = 1.0F - res_28210;
        float B_28212 = x_28211 / 1.0e-2F;
        float x_28213 = B_28212 - res_27968;
        float x_28214 = 4.4999997e-6F * x_28213;
        float A1_28215 = x_28214 / 1.0e-4F;
        float y_28216 = fpow32(B_28212, 2.0F);
        float x_28217 = 1.0000001e-6F * y_28216;
        float A2_28218 = x_28217 / 4.0e-2F;
        float exp_arg_28219 = A1_28215 - A2_28218;
        float res_28220 = fpow32(2.7182817F, exp_arg_28219);
        float negate_arg_28221 = 5.0e-2F * B_28212;
        float exp_arg_28222 = 0.0F - negate_arg_28221;
        float res_28223 = fpow32(2.7182817F, exp_arg_28222);
        float res_28224 = res_28220 * res_28223;
        float res_28225 = res_28207 * res_28224;
        float res_27964 = res_28225 + redout_28361;
        float redout_tmp_28822 = res_27964;
        
        redout_28361 = redout_tmp_28822;
    }
    res_27961 = redout_28361;
    
    float CVA_28226 = 6.0e-3F * res_27961;
    
    scalar_out_28808 = CVA_28226;
    *out_scalar_out_28892 = scalar_out_28808;
    
  cleanup:
    { }
    free(mem_28511);
    free(mem_28513);
    free(mem_28515);
    free(mem_28553);
    free(mem_28555);
    free(mem_28557);
    free(mem_28559);
    free(mem_28610);
    free(mem_28622);
    free(mem_28636);
    free(mem_28665);
    free(mem_28677);
    free(mem_28687);
    free(mem_28689);
    free(mem_28715);
    free(mem_28729);
    free(mem_28743);
    free(mem_28745);
    free(mem_28771);
    free(mem_28785);
    return err;
}
struct futhark_f32_3d {
    struct memblock mem;
    int64_t shape[3];
} ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2)
{
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr =
                          (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) (dim0 * dim1 * dim2) *
                       sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    memmove(arr->mem.mem + 0, data + 0, (size_t) (dim0 * dim1 * dim2) *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2)
{
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr =
                          (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) (dim0 * dim1 * dim2) *
                       sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    memmove(arr->mem.mem + 0, data + offset, (size_t) (dim0 * dim1 * dim2) *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] *
                                                  arr->shape[1] *
                                                  arr->shape[2]) *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_3d(struct futhark_context *ctx,
                                struct futhark_f32_3d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                                    struct futhark_f32_3d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_i64_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const
                                          int64_t *data, int64_t dim0)
{
    struct futhark_i64_1d *bad = NULL;
    struct futhark_i64_1d *arr =
                          (struct futhark_i64_1d *) malloc(sizeof(struct futhark_i64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) dim0 * sizeof(int64_t),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, (size_t) dim0 * sizeof(int64_t));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0)
{
    struct futhark_i64_1d *bad = NULL;
    struct futhark_i64_1d *arr =
                          (struct futhark_i64_1d *) malloc(sizeof(struct futhark_i64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) dim0 * sizeof(int64_t),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, (size_t) dim0 * sizeof(int64_t));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i64_1d(struct futhark_context *ctx,
                          struct futhark_i64_1d *arr, int64_t *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, (size_t) arr->shape[0] *
            sizeof(int64_t));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_i64_1d(struct futhark_context *ctx,
                                struct futhark_i64_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx,
                                    struct futhark_i64_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f32_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, (size_t) dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, (size_t) dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, (size_t) arr->shape[0] * sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                                    struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx, float *out0,
                       struct futhark_f32_3d **out1, const int64_t in0, const
                       int64_t in1, const struct futhark_f32_1d *in2, const
                       struct futhark_i64_1d *in3, const
                       struct futhark_f32_1d *in4, const float in5, const
                       float in6, const float in7, const float in8)
{
    struct memblock swap_term_mem_28510;
    
    swap_term_mem_28510.references = NULL;
    
    struct memblock payments_mem_28511;
    
    payments_mem_28511.references = NULL;
    
    struct memblock notional_mem_28512;
    
    notional_mem_28512.references = NULL;
    
    int64_t n_26622;
    int64_t n_26623;
    int64_t n_26624;
    int64_t paths_26625;
    int64_t steps_26626;
    float a_26630;
    float b_26631;
    float sigma_26632;
    float r0_26633;
    float scalar_out_28808;
    struct memblock out_mem_28809;
    
    out_mem_28809.references = NULL;
    
    int64_t out_arrsizze_28810;
    int64_t out_arrsizze_28811;
    int64_t out_arrsizze_28812;
    
    lock_lock(&ctx->lock);
    paths_26625 = in0;
    steps_26626 = in1;
    swap_term_mem_28510 = in2->mem;
    n_26622 = in2->shape[0];
    payments_mem_28511 = in3->mem;
    n_26623 = in3->shape[0];
    notional_mem_28512 = in4->mem;
    n_26624 = in4->shape[0];
    a_26630 = in5;
    b_26631 = in6;
    sigma_26632 = in7;
    r0_26633 = in8;
    
    int ret = futrts_main(ctx, &scalar_out_28808, &out_mem_28809,
                          &out_arrsizze_28810, &out_arrsizze_28811,
                          &out_arrsizze_28812, swap_term_mem_28510,
                          payments_mem_28511, notional_mem_28512, n_26622,
                          n_26623, n_26624, paths_26625, steps_26626, a_26630,
                          b_26631, sigma_26632, r0_26633);
    
    if (ret == 0) {
        *out0 = scalar_out_28808;
        assert((*out1 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out1)->mem = out_mem_28809;
        (*out1)->shape[0] = out_arrsizze_28810;
        (*out1)->shape[1] = out_arrsizze_28811;
        (*out1)->shape[2] = out_arrsizze_28812;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test(struct futhark_context *ctx, float *out0, const
                       int64_t in0, const int64_t in1)
{
    int64_t paths_27317;
    int64_t steps_27318;
    float scalar_out_28808;
    
    lock_lock(&ctx->lock);
    paths_27317 = in0;
    steps_27318 = in1;
    
    int ret = futrts_test(ctx, &scalar_out_28808, paths_27317, steps_27318);
    
    if (ret == 0) {
        *out0 = scalar_out_28808;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test2(struct futhark_context *ctx, float *out0, const
                        int64_t in0, const int64_t in1, const int64_t in2)
{
    int64_t paths_27744;
    int64_t steps_27745;
    int64_t numswaps_27746;
    float scalar_out_28808;
    
    lock_lock(&ctx->lock);
    paths_27744 = in0;
    steps_27745 = in1;
    numswaps_27746 = in2;
    
    int ret = futrts_test2(ctx, &scalar_out_28808, paths_27744, steps_27745,
                           numswaps_27746);
    
    if (ret == 0) {
        *out0 = scalar_out_28808;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
