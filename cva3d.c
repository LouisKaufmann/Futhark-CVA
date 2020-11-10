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
                       struct futhark_f32_1d **out1, const int64_t in0, const
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
    
    int64_t read_value_27050;
    
    if (read_scalar(&i64_info, &read_value_27050) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_27051;
    
    if (read_scalar(&i64_info, &read_value_27051) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_27052;
    int64_t read_shape_27053[1];
    float *read_arr_27054 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_27054, read_shape_27053, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
                      "[]", f32_info.type_name, strerror(errno));
    
    struct futhark_i64_1d *read_value_27055;
    int64_t read_shape_27056[1];
    int64_t *read_arr_27057 = NULL;
    
    errno = 0;
    if (read_array(&i64_info, (void **) &read_arr_27057, read_shape_27056, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
                      "[]", i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_27058;
    int64_t read_shape_27059[1];
    float *read_arr_27060 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_27060, read_shape_27059, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4,
                      "[]", f32_info.type_name, strerror(errno));
    
    float read_value_27061;
    
    if (read_scalar(&f32_info, &read_value_27061) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      5, f32_info.type_name, strerror(errno));
    
    float read_value_27062;
    
    if (read_scalar(&f32_info, &read_value_27062) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      6, f32_info.type_name, strerror(errno));
    
    float read_value_27063;
    
    if (read_scalar(&f32_info, &read_value_27063) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      7, f32_info.type_name, strerror(errno));
    
    float read_value_27064;
    
    if (read_scalar(&f32_info, &read_value_27064) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      8, f32_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"main\"");
    
    float result_27065;
    struct futhark_f32_1d *result_27066;
    
    if (perform_warmup) {
        int r;
        
        ;
        ;
        assert((read_value_27052 = futhark_new_f32_1d(ctx, read_arr_27054,
                                                      read_shape_27053[0])) !=
            0);
        assert((read_value_27055 = futhark_new_i64_1d(ctx, read_arr_27057,
                                                      read_shape_27056[0])) !=
            0);
        assert((read_value_27058 = futhark_new_f32_1d(ctx, read_arr_27060,
                                                      read_shape_27059[0])) !=
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
        r = futhark_entry_main(ctx, &result_27065, &result_27066,
                               read_value_27050, read_value_27051,
                               read_value_27052, read_value_27055,
                               read_value_27058, read_value_27061,
                               read_value_27062, read_value_27063,
                               read_value_27064);
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
        assert(futhark_free_f32_1d(ctx, read_value_27052) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_27055) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_27058) == 0);
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_f32_1d(ctx, result_27066) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        ;
        assert((read_value_27052 = futhark_new_f32_1d(ctx, read_arr_27054,
                                                      read_shape_27053[0])) !=
            0);
        assert((read_value_27055 = futhark_new_i64_1d(ctx, read_arr_27057,
                                                      read_shape_27056[0])) !=
            0);
        assert((read_value_27058 = futhark_new_f32_1d(ctx, read_arr_27060,
                                                      read_shape_27059[0])) !=
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
        r = futhark_entry_main(ctx, &result_27065, &result_27066,
                               read_value_27050, read_value_27051,
                               read_value_27052, read_value_27055,
                               read_value_27058, read_value_27061,
                               read_value_27062, read_value_27063,
                               read_value_27064);
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
        assert(futhark_free_f32_1d(ctx, read_value_27052) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_27055) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_27058) == 0);
        ;
        ;
        ;
        ;
        if (run < num_runs - 1) {
            ;
            assert(futhark_free_f32_1d(ctx, result_27066) == 0);
        }
    }
    ;
    ;
    free(read_arr_27054);
    free(read_arr_27057);
    free(read_arr_27060);
    ;
    ;
    ;
    ;
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &f32_info, &result_27065);
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_27066)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_27066, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_27066), 1);
        free(arr);
    }
    printf("\n");
    ;
    assert(futhark_free_f32_1d(ctx, result_27066) == 0);
}
static void futrts_cli_entry_test(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    int64_t read_value_27067;
    
    if (read_scalar(&i64_info, &read_value_27067) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_27068;
    
    if (read_scalar(&i64_info, &read_value_27068) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"test\"");
    
    float result_27069;
    
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
        r = futhark_entry_test(ctx, &result_27069, read_value_27067,
                               read_value_27068);
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
        r = futhark_entry_test(ctx, &result_27069, read_value_27067,
                               read_value_27068);
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
    write_scalar(stdout, binary_output, &f32_info, &result_27069);
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
    
    int64_t read_value_27070;
    
    if (read_scalar(&i64_info, &read_value_27070) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_27071;
    
    if (read_scalar(&i64_info, &read_value_27071) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    
    int64_t read_value_27072;
    
    if (read_scalar(&i64_info, &read_value_27072) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      2, i64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"test2\"");
    
    float result_27073;
    
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
        r = futhark_entry_test2(ctx, &result_27073, read_value_27070,
                                read_value_27071, read_value_27072);
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
        r = futhark_entry_test2(ctx, &result_27073, read_value_27070,
                                read_value_27071, read_value_27072);
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
    write_scalar(stdout, binary_output, &f32_info, &result_27073);
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
static float testzistatic_array_realtype_27024[45] = {1.0F, -0.5F, 1.0F, 1.0F,
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
static int64_t testzistatic_array_realtype_27025[45] = {10, 20, 5, 5, 50, 20,
                                                        30, 15, 18, 10, 200, 5,
                                                        5, 50, 20, 30, 15, 18,
                                                        10, 20, 5, 5, 100, 20,
                                                        30, 15, 18, 10, 20, 5,
                                                        5, 50, 20, 30, 15, 18,
                                                        10, 20, 5, 5, 50, 20,
                                                        30, 15, 18};
static float testzistatic_array_realtype_27026[45] = {1.0F, 0.5F, 0.25F, 0.1F,
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
    struct memblock testzistatic_array_26940;
    struct memblock testzistatic_array_26941;
    struct memblock testzistatic_array_26942;
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
    ctx->testzistatic_array_26940 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_27024,
                                                       0};
    ctx->testzistatic_array_26941 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_27025,
                                                       0};
    ctx->testzistatic_array_26942 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_27026,
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
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_26978,
                       struct memblock *out_mem_p_26979,
                       int64_t *out_out_arrsizze_26980,
                       struct memblock swap_term_mem_26561,
                       struct memblock payments_mem_26562,
                       struct memblock notional_mem_26563, int64_t n_24782,
                       int64_t n_24783, int64_t n_24784, int64_t paths_24785,
                       int64_t steps_24786, float a_24790, float b_24791,
                       float sigma_24792, float r0_24793);
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_27001,
                       int64_t paths_25265, int64_t steps_25266);
static int futrts_test2(struct futhark_context *ctx,
                        float *out_scalar_out_27027, int64_t paths_25705,
                        int64_t steps_25706, int64_t numswaps_25707);
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
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_26978,
                       struct memblock *out_mem_p_26979,
                       int64_t *out_out_arrsizze_26980,
                       struct memblock swap_term_mem_26561,
                       struct memblock payments_mem_26562,
                       struct memblock notional_mem_26563, int64_t n_24782,
                       int64_t n_24783, int64_t n_24784, int64_t paths_24785,
                       int64_t steps_24786, float a_24790, float b_24791,
                       float sigma_24792, float r0_24793)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_26565_cached_sizze_26981 = 0;
    char *mem_26565 = NULL;
    size_t mem_26567_cached_sizze_26982 = 0;
    char *mem_26567 = NULL;
    size_t mem_26569_cached_sizze_26983 = 0;
    char *mem_26569 = NULL;
    size_t mem_26571_cached_sizze_26984 = 0;
    char *mem_26571 = NULL;
    size_t mem_26621_cached_sizze_26985 = 0;
    char *mem_26621 = NULL;
    size_t mem_26637_cached_sizze_26986 = 0;
    char *mem_26637 = NULL;
    size_t mem_26641_cached_sizze_26987 = 0;
    char *mem_26641 = NULL;
    size_t mem_26671_cached_sizze_26988 = 0;
    char *mem_26671 = NULL;
    size_t mem_26685_cached_sizze_26989 = 0;
    char *mem_26685 = NULL;
    size_t mem_26727_cached_sizze_26990 = 0;
    char *mem_26727 = NULL;
    size_t mem_26729_cached_sizze_26991 = 0;
    char *mem_26729 = NULL;
    size_t mem_26781_cached_sizze_26992 = 0;
    char *mem_26781 = NULL;
    size_t mem_26783_cached_sizze_26993 = 0;
    char *mem_26783 = NULL;
    size_t mem_26809_cached_sizze_26994 = 0;
    char *mem_26809 = NULL;
    size_t mem_26823_cached_sizze_26995 = 0;
    char *mem_26823 = NULL;
    size_t mem_26837_cached_sizze_26996 = 0;
    char *mem_26837 = NULL;
    size_t mem_26839_cached_sizze_26997 = 0;
    char *mem_26839 = NULL;
    size_t mem_26865_cached_sizze_26998 = 0;
    char *mem_26865 = NULL;
    size_t mem_26879_cached_sizze_26999 = 0;
    char *mem_26879 = NULL;
    size_t mem_26893_cached_sizze_27000 = 0;
    char *mem_26893 = NULL;
    float scalar_out_26939;
    struct memblock out_mem_26940;
    
    out_mem_26940.references = NULL;
    
    int64_t out_arrsizze_26941;
    bool dim_match_24794 = n_24782 == n_24783;
    bool empty_or_match_cert_24795;
    
    if (!dim_match_24794) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cva3d.fut:101:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_24796 = n_24782 == n_24784;
    bool empty_or_match_cert_24797;
    
    if (!dim_match_24796) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cva3d.fut:101:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
            return 1;
        return 1;
    }
    
    float res_24799;
    float redout_26201 = -INFINITY;
    
    for (int64_t i_26202 = 0; i_26202 < n_24782; i_26202++) {
        float x_24803 = ((float *) swap_term_mem_26561.mem)[i_26202];
        int64_t x_24804 = ((int64_t *) payments_mem_26562.mem)[i_26202];
        float res_24805 = sitofp_i64_f32(x_24804);
        float res_24806 = x_24803 * res_24805;
        float res_24802 = fmax32(res_24806, redout_26201);
        float redout_tmp_26942 = res_24802;
        
        redout_26201 = redout_tmp_26942;
    }
    res_24799 = redout_26201;
    
    float res_24807 = sitofp_i64_f32(steps_24786);
    float dt_24808 = res_24799 / res_24807;
    float x_24810 = fpow32(a_24790, 2.0F);
    float x_24811 = b_24791 * x_24810;
    float x_24812 = fpow32(sigma_24792, 2.0F);
    float y_24813 = x_24812 / 2.0F;
    float y_24814 = x_24811 - y_24813;
    float y_24815 = 4.0F * a_24790;
    int64_t bytes_26564 = 4 * n_24782;
    
    if (mem_26565_cached_sizze_26981 < (size_t) bytes_26564) {
        mem_26565 = realloc(mem_26565, bytes_26564);
        mem_26565_cached_sizze_26981 = bytes_26564;
    }
    if (mem_26567_cached_sizze_26982 < (size_t) bytes_26564) {
        mem_26567 = realloc(mem_26567, bytes_26564);
        mem_26567_cached_sizze_26982 = bytes_26564;
    }
    
    int64_t bytes_26568 = 8 * n_24782;
    
    if (mem_26569_cached_sizze_26983 < (size_t) bytes_26568) {
        mem_26569 = realloc(mem_26569, bytes_26568);
        mem_26569_cached_sizze_26983 = bytes_26568;
    }
    if (mem_26571_cached_sizze_26984 < (size_t) bytes_26564) {
        mem_26571 = realloc(mem_26571, bytes_26564);
        mem_26571_cached_sizze_26984 = bytes_26564;
    }
    for (int64_t i_26213 = 0; i_26213 < n_24782; i_26213++) {
        float res_24825 = ((float *) swap_term_mem_26561.mem)[i_26213];
        int64_t res_24826 = ((int64_t *) payments_mem_26562.mem)[i_26213];
        bool bounds_invalid_upwards_24828 = slt64(res_24826, 1);
        bool valid_24829 = !bounds_invalid_upwards_24828;
        bool range_valid_c_24830;
        
        if (!valid_24829) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_24826,
                          " is invalid.",
                          "-> #0  cva3d.fut:55:29-48\n   #1  cva3d.fut:96:25-65\n   #2  cva3d.fut:110:16-62\n   #3  cva3d.fut:106:17-110:85\n   #4  cva3d.fut:101:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
                return 1;
            return 1;
        }
        
        bool y_24832 = slt64(0, res_24826);
        bool index_certs_24833;
        
        if (!y_24832) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_24826, "].",
                                   "-> #0  cva3d.fut:97:47-70\n   #1  cva3d.fut:110:16-62\n   #2  cva3d.fut:106:17-110:85\n   #3  cva3d.fut:101:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
                return 1;
            return 1;
        }
        
        float binop_y_24834 = sitofp_i64_f32(res_24826);
        float index_primexp_24835 = res_24825 * binop_y_24834;
        float negate_arg_24836 = a_24790 * index_primexp_24835;
        float exp_arg_24837 = 0.0F - negate_arg_24836;
        float res_24838 = fpow32(2.7182817F, exp_arg_24837);
        float x_24839 = 1.0F - res_24838;
        float B_24840 = x_24839 / a_24790;
        float x_24841 = B_24840 - index_primexp_24835;
        float x_24842 = y_24814 * x_24841;
        float A1_24843 = x_24842 / x_24810;
        float y_24844 = fpow32(B_24840, 2.0F);
        float x_24845 = x_24812 * y_24844;
        float A2_24846 = x_24845 / y_24815;
        float exp_arg_24847 = A1_24843 - A2_24846;
        float res_24848 = fpow32(2.7182817F, exp_arg_24847);
        float negate_arg_24849 = r0_24793 * B_24840;
        float exp_arg_24850 = 0.0F - negate_arg_24849;
        float res_24851 = fpow32(2.7182817F, exp_arg_24850);
        float res_24852 = res_24848 * res_24851;
        float res_24853;
        float redout_26203 = 0.0F;
        
        for (int64_t i_26204 = 0; i_26204 < res_24826; i_26204++) {
            int64_t index_primexp_26348 = add64(1, i_26204);
            float res_24858 = sitofp_i64_f32(index_primexp_26348);
            float res_24859 = res_24825 * res_24858;
            float negate_arg_24860 = a_24790 * res_24859;
            float exp_arg_24861 = 0.0F - negate_arg_24860;
            float res_24862 = fpow32(2.7182817F, exp_arg_24861);
            float x_24863 = 1.0F - res_24862;
            float B_24864 = x_24863 / a_24790;
            float x_24865 = B_24864 - res_24859;
            float x_24866 = y_24814 * x_24865;
            float A1_24867 = x_24866 / x_24810;
            float y_24868 = fpow32(B_24864, 2.0F);
            float x_24869 = x_24812 * y_24868;
            float A2_24870 = x_24869 / y_24815;
            float exp_arg_24871 = A1_24867 - A2_24870;
            float res_24872 = fpow32(2.7182817F, exp_arg_24871);
            float negate_arg_24873 = r0_24793 * B_24864;
            float exp_arg_24874 = 0.0F - negate_arg_24873;
            float res_24875 = fpow32(2.7182817F, exp_arg_24874);
            float res_24876 = res_24872 * res_24875;
            float res_24856 = res_24876 + redout_26203;
            float redout_tmp_26947 = res_24856;
            
            redout_26203 = redout_tmp_26947;
        }
        res_24853 = redout_26203;
        
        float x_24877 = 1.0F - res_24852;
        float y_24878 = res_24825 * res_24853;
        float res_24879 = x_24877 / y_24878;
        
        ((float *) mem_26565)[i_26213] = res_24879;
        memmove(mem_26567 + i_26213 * 4, notional_mem_26563.mem + i_26213 * 4,
                (int32_t) sizeof(float));
        memmove(mem_26569 + i_26213 * 8, payments_mem_26562.mem + i_26213 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_26571 + i_26213 * 4, swap_term_mem_26561.mem + i_26213 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_24880 = res_24807 / res_24799;
    bool bounds_invalid_upwards_24881 = slt64(steps_24786, 1);
    bool valid_24882 = !bounds_invalid_upwards_24881;
    bool range_valid_c_24883;
    
    if (!valid_24882) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_24786,
                               " is invalid.",
                               "-> #0  cva3d.fut:61:56-67\n   #1  cva3d.fut:112:17-44\n   #2  cva3d.fut:101:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_26620 = 4 * steps_24786;
    
    if (mem_26621_cached_sizze_26985 < (size_t) bytes_26620) {
        mem_26621 = realloc(mem_26621, bytes_26620);
        mem_26621_cached_sizze_26985 = bytes_26620;
    }
    for (int64_t i_26220 = 0; i_26220 < steps_24786; i_26220++) {
        int64_t index_primexp_26355 = add64(1, i_26220);
        float res_24887 = sitofp_i64_f32(index_primexp_26355);
        float res_24888 = res_24887 / sims_per_year_24880;
        
        ((float *) mem_26621)[i_26220] = res_24888;
    }
    
    bool bounds_invalid_upwards_24889 = slt64(paths_24785, 0);
    bool valid_24890 = !bounds_invalid_upwards_24889;
    bool range_valid_c_24891;
    
    if (!valid_24890) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_24785,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva3d.fut:116:19-49\n   #4  cva3d.fut:101:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
            return 1;
        return 1;
    }
    
    int64_t upper_bound_24894 = sub64(steps_24786, 1);
    float res_24895;
    
    res_24895 = futrts_sqrt32(dt_24808);
    
    int64_t binop_x_26635 = paths_24785 * steps_24786;
    int64_t binop_x_26636 = n_24782 * binop_x_26635;
    int64_t bytes_26634 = 4 * binop_x_26636;
    
    if (mem_26637_cached_sizze_26986 < (size_t) bytes_26634) {
        mem_26637 = realloc(mem_26637, bytes_26634);
        mem_26637_cached_sizze_26986 = bytes_26634;
    }
    if (mem_26641_cached_sizze_26987 < (size_t) bytes_26634) {
        mem_26641 = realloc(mem_26641, bytes_26634);
        mem_26641_cached_sizze_26987 = bytes_26634;
    }
    
    int64_t ctx_val_26665 = n_24782 * steps_24786;
    
    if (mem_26671_cached_sizze_26988 < (size_t) bytes_26620) {
        mem_26671 = realloc(mem_26671, bytes_26620);
        mem_26671_cached_sizze_26988 = bytes_26620;
    }
    if (mem_26685_cached_sizze_26989 < (size_t) bytes_26620) {
        mem_26685 = realloc(mem_26685, bytes_26620);
        mem_26685_cached_sizze_26989 = bytes_26620;
    }
    if (mem_26727_cached_sizze_26990 < (size_t) bytes_26564) {
        mem_26727 = realloc(mem_26727, bytes_26564);
        mem_26727_cached_sizze_26990 = bytes_26564;
    }
    if (mem_26729_cached_sizze_26991 < (size_t) bytes_26564) {
        mem_26729 = realloc(mem_26729, bytes_26564);
        mem_26729_cached_sizze_26991 = bytes_26564;
    }
    for (int64_t i_26237 = 0; i_26237 < paths_24785; i_26237++) {
        int32_t res_24899 = sext_i64_i32(i_26237);
        int32_t x_24900 = lshr32(res_24899, 16);
        int32_t x_24901 = res_24899 ^ x_24900;
        int32_t x_24902 = mul32(73244475, x_24901);
        int32_t x_24903 = lshr32(x_24902, 16);
        int32_t x_24904 = x_24902 ^ x_24903;
        int32_t x_24905 = mul32(73244475, x_24904);
        int32_t x_24906 = lshr32(x_24905, 16);
        int32_t x_24907 = x_24905 ^ x_24906;
        int32_t unsign_arg_24908 = 777822902 ^ x_24907;
        int32_t unsign_arg_24909 = mul32(48271, unsign_arg_24908);
        int32_t unsign_arg_24910 = umod32(unsign_arg_24909, 2147483647);
        
        for (int64_t i_26224 = 0; i_26224 < steps_24786; i_26224++) {
            int32_t res_24913 = sext_i64_i32(i_26224);
            int32_t x_24914 = lshr32(res_24913, 16);
            int32_t x_24915 = res_24913 ^ x_24914;
            int32_t x_24916 = mul32(73244475, x_24915);
            int32_t x_24917 = lshr32(x_24916, 16);
            int32_t x_24918 = x_24916 ^ x_24917;
            int32_t x_24919 = mul32(73244475, x_24918);
            int32_t x_24920 = lshr32(x_24919, 16);
            int32_t x_24921 = x_24919 ^ x_24920;
            int32_t unsign_arg_24922 = unsign_arg_24910 ^ x_24921;
            int32_t unsign_arg_24923 = mul32(48271, unsign_arg_24922);
            int32_t unsign_arg_24924 = umod32(unsign_arg_24923, 2147483647);
            int32_t unsign_arg_24925 = mul32(48271, unsign_arg_24924);
            int32_t unsign_arg_24926 = umod32(unsign_arg_24925, 2147483647);
            float res_24927 = uitofp_i32_f32(unsign_arg_24924);
            float res_24928 = res_24927 / 2.1474836e9F;
            float res_24929 = uitofp_i32_f32(unsign_arg_24926);
            float res_24930 = res_24929 / 2.1474836e9F;
            float res_24931;
            
            res_24931 = futrts_log32(res_24928);
            
            float res_24932 = -2.0F * res_24931;
            float res_24933;
            
            res_24933 = futrts_sqrt32(res_24932);
            
            float res_24934 = 6.2831855F * res_24930;
            float res_24935;
            
            res_24935 = futrts_cos32(res_24934);
            
            float res_24936 = res_24933 * res_24935;
            
            ((float *) mem_26671)[i_26224] = res_24936;
        }
        for (int64_t i_26952 = 0; i_26952 < steps_24786; i_26952++) {
            ((float *) mem_26685)[i_26952] = r0_24793;
        }
        for (int64_t i_24939 = 0; i_24939 < upper_bound_24894; i_24939++) {
            bool y_24941 = slt64(i_24939, steps_24786);
            bool index_certs_24942;
            
            if (!y_24941) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_24939,
                              "] out of bounds for array of shape [",
                              steps_24786, "].",
                              "-> #0  cva3d.fut:72:97-104\n   #1  cva3d.fut:124:32-62\n   #2  cva3d.fut:124:22-69\n   #3  cva3d.fut:101:1-147:20\n");
                if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
                    return 1;
                return 1;
            }
            
            float shortstep_arg_24943 = ((float *) mem_26671)[i_24939];
            float shortstep_arg_24944 = ((float *) mem_26685)[i_24939];
            float y_24945 = b_24791 - shortstep_arg_24944;
            float x_24946 = a_24790 * y_24945;
            float x_24947 = dt_24808 * x_24946;
            float x_24948 = res_24895 * shortstep_arg_24943;
            float y_24949 = sigma_24792 * x_24948;
            float delta_r_24950 = x_24947 + y_24949;
            float res_24951 = shortstep_arg_24944 + delta_r_24950;
            int64_t i_24952 = add64(1, i_24939);
            bool x_24953 = sle64(0, i_24952);
            bool y_24954 = slt64(i_24952, steps_24786);
            bool bounds_check_24955 = x_24953 && y_24954;
            bool index_certs_24956;
            
            if (!bounds_check_24955) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_24952,
                              "] out of bounds for array of shape [",
                              steps_24786, "].",
                              "-> #0  cva3d.fut:72:58-105\n   #1  cva3d.fut:124:32-62\n   #2  cva3d.fut:124:22-69\n   #3  cva3d.fut:101:1-147:20\n");
                if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
                    return 1;
                return 1;
            }
            ((float *) mem_26685)[i_24952] = res_24951;
        }
        for (int64_t i_26230 = 0; i_26230 < steps_24786; i_26230++) {
            float x_24960 = ((float *) mem_26685)[i_26230];
            float x_24961 = ((float *) mem_26621)[i_26230];
            
            for (int64_t i_26956 = 0; i_26956 < n_24782; i_26956++) {
                ((float *) mem_26727)[i_26956] = x_24960;
            }
            for (int64_t i_26957 = 0; i_26957 < n_24782; i_26957++) {
                ((float *) mem_26729)[i_26957] = x_24961;
            }
            memmove(mem_26637 + (i_26237 * ctx_val_26665 + i_26230 * n_24782) *
                    4, mem_26727 + 0, n_24782 * (int32_t) sizeof(float));
            memmove(mem_26641 + (i_26237 * ctx_val_26665 + i_26230 * n_24782) *
                    4, mem_26729 + 0, n_24782 * (int32_t) sizeof(float));
        }
    }
    
    int64_t flat_dim_24965 = n_24782 * binop_x_26635;
    int64_t bytes_26780 = 8 * flat_dim_24965;
    
    if (mem_26781_cached_sizze_26992 < (size_t) bytes_26780) {
        mem_26781 = realloc(mem_26781, bytes_26780);
        mem_26781_cached_sizze_26992 = bytes_26780;
    }
    if (mem_26783_cached_sizze_26993 < (size_t) bytes_26780) {
        mem_26783 = realloc(mem_26783, bytes_26780);
        mem_26783_cached_sizze_26993 = bytes_26780;
    }
    
    int64_t discard_26248;
    int64_t scanacc_26242 = 0;
    
    for (int64_t i_26245 = 0; i_26245 < flat_dim_24965; i_26245++) {
        int64_t binop_x_26369 = squot64(i_26245, ctx_val_26665);
        int64_t binop_y_26371 = binop_x_26369 * ctx_val_26665;
        int64_t binop_x_26372 = i_26245 - binop_y_26371;
        int64_t binop_x_26378 = squot64(binop_x_26372, n_24782);
        int64_t binop_y_26379 = n_24782 * binop_x_26378;
        int64_t new_index_26380 = binop_x_26372 - binop_y_26379;
        int64_t x_24976 = ((int64_t *) mem_26569)[new_index_26380];
        float x_24977 = ((float *) mem_26571)[new_index_26380];
        float x_24978 = ((float *) mem_26641)[binop_x_26369 * ctx_val_26665 +
                                              binop_x_26378 * n_24782 +
                                              new_index_26380];
        float x_24979 = x_24978 / x_24977;
        float ceil_arg_24980 = x_24979 - 1.0F;
        float res_24981;
        
        res_24981 = futrts_ceil32(ceil_arg_24980);
        
        int64_t res_24982 = fptosi_f32_i64(res_24981);
        int64_t max_arg_24983 = sub64(x_24976, res_24982);
        int64_t res_24984 = smax64(0, max_arg_24983);
        bool cond_24985 = res_24984 == 0;
        int64_t res_24986;
        
        if (cond_24985) {
            res_24986 = 1;
        } else {
            res_24986 = res_24984;
        }
        
        int64_t res_24975 = add64(res_24986, scanacc_26242);
        
        ((int64_t *) mem_26781)[i_26245] = res_24975;
        ((int64_t *) mem_26783)[i_26245] = res_24986;
        
        int64_t scanacc_tmp_26958 = res_24975;
        
        scanacc_26242 = scanacc_tmp_26958;
    }
    discard_26248 = scanacc_26242;
    
    int64_t res_24989;
    int64_t redout_26249 = 0;
    
    for (int64_t i_26250 = 0; i_26250 < flat_dim_24965; i_26250++) {
        int64_t x_24993 = ((int64_t *) mem_26783)[i_26250];
        int64_t res_24992 = add64(x_24993, redout_26249);
        int64_t redout_tmp_26961 = res_24992;
        
        redout_26249 = redout_tmp_26961;
    }
    res_24989 = redout_26249;
    
    bool bounds_invalid_upwards_24994 = slt64(res_24989, 0);
    bool valid_24995 = !bounds_invalid_upwards_24994;
    bool range_valid_c_24996;
    
    if (!valid_24995) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", res_24989,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:101:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_26808 = 8 * res_24989;
    
    if (mem_26809_cached_sizze_26994 < (size_t) bytes_26808) {
        mem_26809 = realloc(mem_26809, bytes_26808);
        mem_26809_cached_sizze_26994 = bytes_26808;
    }
    for (int64_t i_26962 = 0; i_26962 < res_24989; i_26962++) {
        ((int64_t *) mem_26809)[i_26962] = 0;
    }
    for (int64_t iter_26251 = 0; iter_26251 < flat_dim_24965; iter_26251++) {
        int64_t i_p_o_26425 = add64(-1, iter_26251);
        int64_t rot_i_26426 = smod64(i_p_o_26425, flat_dim_24965);
        int64_t pixel_26254 = ((int64_t *) mem_26781)[rot_i_26426];
        bool cond_25004 = iter_26251 == 0;
        int64_t res_25005;
        
        if (cond_25004) {
            res_25005 = 0;
        } else {
            res_25005 = pixel_26254;
        }
        
        bool less_than_zzero_26255 = slt64(res_25005, 0);
        bool greater_than_sizze_26256 = sle64(res_24989, res_25005);
        bool outside_bounds_dim_26257 = less_than_zzero_26255 ||
             greater_than_sizze_26256;
        
        if (!outside_bounds_dim_26257) {
            int64_t read_hist_26259 = ((int64_t *) mem_26809)[res_25005];
            int64_t res_25001 = smax64(iter_26251, read_hist_26259);
            
            ((int64_t *) mem_26809)[res_25005] = res_25001;
        }
    }
    if (mem_26823_cached_sizze_26995 < (size_t) bytes_26808) {
        mem_26823 = realloc(mem_26823, bytes_26808);
        mem_26823_cached_sizze_26995 = bytes_26808;
    }
    
    int64_t discard_26272;
    int64_t scanacc_26265 = 0;
    
    for (int64_t i_26268 = 0; i_26268 < res_24989; i_26268++) {
        int64_t x_25015 = ((int64_t *) mem_26809)[i_26268];
        bool res_25016 = slt64(0, x_25015);
        int64_t res_25013;
        
        if (res_25016) {
            res_25013 = x_25015;
        } else {
            int64_t res_25014 = add64(x_25015, scanacc_26265);
            
            res_25013 = res_25014;
        }
        ((int64_t *) mem_26823)[i_26268] = res_25013;
        
        int64_t scanacc_tmp_26964 = res_25013;
        
        scanacc_26265 = scanacc_tmp_26964;
    }
    discard_26272 = scanacc_26265;
    
    int64_t bytes_26836 = 4 * res_24989;
    
    if (mem_26837_cached_sizze_26996 < (size_t) bytes_26836) {
        mem_26837 = realloc(mem_26837, bytes_26836);
        mem_26837_cached_sizze_26996 = bytes_26836;
    }
    if (mem_26839_cached_sizze_26997 < (size_t) res_24989) {
        mem_26839 = realloc(mem_26839, res_24989);
        mem_26839_cached_sizze_26997 = res_24989;
    }
    
    int64_t lstel_tmp_25068 = 1;
    int64_t inpacc_25026;
    float inpacc_25028;
    int64_t inpacc_25034;
    float inpacc_25036;
    
    inpacc_25034 = 0;
    inpacc_25036 = 0.0F;
    for (int64_t i_26311 = 0; i_26311 < res_24989; i_26311++) {
        int64_t x_26440 = ((int64_t *) mem_26823)[i_26311];
        int64_t i_p_o_26442 = add64(-1, i_26311);
        int64_t rot_i_26443 = smod64(i_p_o_26442, res_24989);
        int64_t x_26444 = ((int64_t *) mem_26823)[rot_i_26443];
        bool res_26445 = x_26440 == x_26444;
        bool res_26446 = !res_26445;
        int64_t res_25070;
        
        if (res_26446) {
            res_25070 = lstel_tmp_25068;
        } else {
            int64_t res_25071 = add64(1, inpacc_25034);
            
            res_25070 = res_25071;
        }
        
        int64_t res_25085;
        
        if (res_26446) {
            res_25085 = 1;
        } else {
            int64_t res_25086 = add64(1, inpacc_25034);
            
            res_25085 = res_25086;
        }
        
        int64_t res_25087 = sub64(res_25085, 1);
        bool x_26461 = sle64(0, x_26440);
        bool y_26462 = slt64(x_26440, flat_dim_24965);
        bool bounds_check_26463 = x_26461 && y_26462;
        bool index_certs_26464;
        
        if (!bounds_check_26463) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", x_26440,
                                   "] out of bounds for array of shape [",
                                   flat_dim_24965, "].",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:101:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
                return 1;
            return 1;
        }
        
        int64_t new_index_26465 = squot64(x_26440, ctx_val_26665);
        int64_t binop_y_26466 = new_index_26465 * ctx_val_26665;
        int64_t binop_x_26467 = x_26440 - binop_y_26466;
        int64_t new_index_26468 = squot64(binop_x_26467, n_24782);
        int64_t binop_y_26469 = n_24782 * new_index_26468;
        int64_t new_index_26470 = binop_x_26467 - binop_y_26469;
        float lifted_0_get_arg_26471 = ((float *) mem_26637)[new_index_26465 *
                                                             ctx_val_26665 +
                                                             new_index_26468 *
                                                             n_24782 +
                                                             new_index_26470];
        float lifted_0_get_arg_26472 = ((float *) mem_26565)[new_index_26470];
        float lifted_0_get_arg_26473 = ((float *) mem_26567)[new_index_26470];
        int64_t lifted_0_get_arg_26474 =
                ((int64_t *) mem_26569)[new_index_26470];
        float lifted_0_get_arg_26475 = ((float *) mem_26571)[new_index_26470];
        float lifted_0_get_arg_26476 = ((float *) mem_26641)[new_index_26465 *
                                                             ctx_val_26665 +
                                                             new_index_26468 *
                                                             n_24782 +
                                                             new_index_26470];
        float x_26477 = lifted_0_get_arg_26476 / lifted_0_get_arg_26475;
        float ceil_arg_26478 = x_26477 - 1.0F;
        float res_26479;
        
        res_26479 = futrts_ceil32(ceil_arg_26478);
        
        int64_t res_26480 = fptosi_f32_i64(res_26479);
        int64_t max_arg_26481 = sub64(lifted_0_get_arg_26474, res_26480);
        int64_t res_26482 = smax64(0, max_arg_26481);
        bool cond_26483 = res_26482 == 0;
        float res_26484;
        
        if (cond_26483) {
            res_26484 = 0.0F;
        } else {
            float res_26485;
            
            res_26485 = futrts_ceil32(x_26477);
            
            float start_26486 = lifted_0_get_arg_26475 * res_26485;
            float res_26487;
            
            res_26487 = futrts_ceil32(ceil_arg_26478);
            
            int64_t res_26488 = fptosi_f32_i64(res_26487);
            int64_t max_arg_26489 = sub64(lifted_0_get_arg_26474, res_26488);
            int64_t res_26490 = smax64(0, max_arg_26489);
            int64_t sizze_26491 = sub64(res_26490, 1);
            bool cond_26492 = res_25087 == 0;
            float res_26493;
            
            if (cond_26492) {
                res_26493 = 1.0F;
            } else {
                res_26493 = 0.0F;
            }
            
            bool cond_26494 = slt64(0, res_25087);
            float res_26495;
            
            if (cond_26494) {
                float y_26496 = lifted_0_get_arg_26472 * lifted_0_get_arg_26475;
                float res_26497 = res_26493 - y_26496;
                
                res_26495 = res_26497;
            } else {
                res_26495 = res_26493;
            }
            
            bool cond_26498 = res_25087 == sizze_26491;
            float res_26499;
            
            if (cond_26498) {
                float res_26500 = res_26495 - 1.0F;
                
                res_26499 = res_26500;
            } else {
                res_26499 = res_26495;
            }
            
            float res_26501 = lifted_0_get_arg_26473 * res_26499;
            float res_26502 = sitofp_i64_f32(res_25087);
            float y_26503 = lifted_0_get_arg_26475 * res_26502;
            float bondprice_arg_26504 = start_26486 + y_26503;
            float y_26505 = bondprice_arg_26504 - lifted_0_get_arg_26476;
            float negate_arg_26506 = a_24790 * y_26505;
            float exp_arg_26507 = 0.0F - negate_arg_26506;
            float res_26508 = fpow32(2.7182817F, exp_arg_26507);
            float x_26509 = 1.0F - res_26508;
            float B_26510 = x_26509 / a_24790;
            float x_26511 = B_26510 - bondprice_arg_26504;
            float x_26512 = lifted_0_get_arg_26476 + x_26511;
            float x_26518 = y_24814 * x_26512;
            float A1_26519 = x_26518 / x_24810;
            float y_26520 = fpow32(B_26510, 2.0F);
            float x_26521 = x_24812 * y_26520;
            float A2_26523 = x_26521 / y_24815;
            float exp_arg_26524 = A1_26519 - A2_26523;
            float res_26525 = fpow32(2.7182817F, exp_arg_26524);
            float negate_arg_26526 = lifted_0_get_arg_26471 * B_26510;
            float exp_arg_26527 = 0.0F - negate_arg_26526;
            float res_26528 = fpow32(2.7182817F, exp_arg_26527);
            float res_26529 = res_26525 * res_26528;
            float res_26530 = res_26501 * res_26529;
            
            res_26484 = res_26530;
        }
        
        float lstel_tmp_25161 = res_26484;
        float res_25167;
        
        if (res_26446) {
            res_25167 = res_26484;
        } else {
            float res_25168 = inpacc_25036 + res_26484;
            
            res_25167 = res_25168;
        }
        
        float res_25170;
        
        if (res_26446) {
            res_25170 = lstel_tmp_25161;
        } else {
            float res_25171 = inpacc_25036 + res_26484;
            
            res_25170 = res_25171;
        }
        ((float *) mem_26837)[i_26311] = res_25167;
        ((bool *) mem_26839)[i_26311] = res_26446;
        
        int64_t inpacc_tmp_26966 = res_25070;
        float inpacc_tmp_26967 = res_25170;
        
        inpacc_25034 = inpacc_tmp_26966;
        inpacc_25036 = inpacc_tmp_26967;
    }
    inpacc_25026 = inpacc_25034;
    inpacc_25028 = inpacc_25036;
    if (mem_26865_cached_sizze_26998 < (size_t) bytes_26808) {
        mem_26865 = realloc(mem_26865, bytes_26808);
        mem_26865_cached_sizze_26998 = bytes_26808;
    }
    
    int64_t discard_26320;
    int64_t scanacc_26316 = 0;
    
    for (int64_t i_26318 = 0; i_26318 < res_24989; i_26318++) {
        int64_t i_p_o_26541 = add64(1, i_26318);
        int64_t rot_i_26542 = smod64(i_p_o_26541, res_24989);
        bool x_25184 = ((bool *) mem_26839)[rot_i_26542];
        int64_t res_25185 = btoi_bool_i64(x_25184);
        int64_t res_25183 = add64(res_25185, scanacc_26316);
        
        ((int64_t *) mem_26865)[i_26318] = res_25183;
        
        int64_t scanacc_tmp_26970 = res_25183;
        
        scanacc_26316 = scanacc_tmp_26970;
    }
    discard_26320 = scanacc_26316;
    
    bool cond_25186 = slt64(0, res_24989);
    int64_t num_segments_25187;
    
    if (cond_25186) {
        int64_t i_25188 = sub64(res_24989, 1);
        bool x_25189 = sle64(0, i_25188);
        bool y_25190 = slt64(i_25188, res_24989);
        bool bounds_check_25191 = x_25189 && y_25190;
        bool index_certs_25192;
        
        if (!bounds_check_25191) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", i_25188,
                                   "] out of bounds for array of shape [",
                                   res_24989, "].",
                                   "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva3d.fut:135:18-77\n   #5  cva3d.fut:101:1-147:20\n");
            if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
                return 1;
            return 1;
        }
        
        int64_t res_25193 = ((int64_t *) mem_26865)[i_25188];
        
        num_segments_25187 = res_25193;
    } else {
        num_segments_25187 = 0;
    }
    
    bool bounds_invalid_upwards_25194 = slt64(num_segments_25187, 0);
    bool valid_25195 = !bounds_invalid_upwards_25194;
    bool range_valid_c_25196;
    
    if (!valid_25195) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", num_segments_25187,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:101:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_26878 = 4 * num_segments_25187;
    
    if (mem_26879_cached_sizze_26999 < (size_t) bytes_26878) {
        mem_26879 = realloc(mem_26879, bytes_26878);
        mem_26879_cached_sizze_26999 = bytes_26878;
    }
    for (int64_t i_26972 = 0; i_26972 < num_segments_25187; i_26972++) {
        ((float *) mem_26879)[i_26972] = 0.0F;
    }
    for (int64_t write_iter_26321 = 0; write_iter_26321 < res_24989;
         write_iter_26321++) {
        int64_t write_iv_26323 = ((int64_t *) mem_26865)[write_iter_26321];
        int64_t i_p_o_26544 = add64(1, write_iter_26321);
        int64_t rot_i_26545 = smod64(i_p_o_26544, res_24989);
        bool write_iv_26324 = ((bool *) mem_26839)[rot_i_26545];
        int64_t res_25202;
        
        if (write_iv_26324) {
            int64_t res_25203 = sub64(write_iv_26323, 1);
            
            res_25202 = res_25203;
        } else {
            res_25202 = -1;
        }
        
        bool less_than_zzero_26326 = slt64(res_25202, 0);
        bool greater_than_sizze_26327 = sle64(num_segments_25187, res_25202);
        bool outside_bounds_dim_26328 = less_than_zzero_26326 ||
             greater_than_sizze_26327;
        
        if (!outside_bounds_dim_26328) {
            memmove(mem_26879 + res_25202 * 4, mem_26837 + write_iter_26321 * 4,
                    (int32_t) sizeof(float));
        }
    }
    
    bool dim_match_25204 = flat_dim_24965 == num_segments_25187;
    bool empty_or_match_cert_25205;
    
    if (!dim_match_25204) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Value of (core language) shape (",
                               num_segments_25187,
                               ") cannot match shape of type `[",
                               flat_dim_24965, "]b`.",
                               "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva3d.fut:135:18-77\n   #2  cva3d.fut:101:1-147:20\n");
        if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
            return 1;
        return 1;
    }
    
    float res_25208 = sitofp_i64_f32(paths_24785);
    
    if (mem_26893_cached_sizze_27000 < (size_t) bytes_26620) {
        mem_26893 = realloc(mem_26893, bytes_26620);
        mem_26893_cached_sizze_27000 = bytes_26620;
    }
    
    float res_25225;
    float redout_26337 = 0.0F;
    
    for (int64_t i_26339 = 0; i_26339 < steps_24786; i_26339++) {
        float x_25231 = ((float *) mem_26621)[i_26339];
        int64_t binop_y_26548 = n_24782 * i_26339;
        float res_25232;
        float redout_26334 = 0.0F;
        
        for (int64_t i_26335 = 0; i_26335 < paths_24785; i_26335++) {
            int64_t binop_x_26547 = i_26335 * ctx_val_26665;
            int64_t binop_x_26549 = binop_x_26547 + binop_y_26548;
            float res_25237;
            float redout_26332 = 0.0F;
            
            for (int64_t i_26333 = 0; i_26333 < n_24782; i_26333++) {
                int64_t new_index_26550 = i_26333 + binop_x_26549;
                float x_25241 = ((float *) mem_26879)[new_index_26550];
                float res_25240 = x_25241 + redout_26332;
                float redout_tmp_26977 = res_25240;
                
                redout_26332 = redout_tmp_26977;
            }
            res_25237 = redout_26332;
            
            float res_25242 = fmax32(0.0F, res_25237);
            float res_25235 = res_25242 + redout_26334;
            float redout_tmp_26976 = res_25235;
            
            redout_26334 = redout_tmp_26976;
        }
        res_25232 = redout_26334;
        
        float res_25243 = res_25232 / res_25208;
        float negate_arg_25244 = a_24790 * x_25231;
        float exp_arg_25245 = 0.0F - negate_arg_25244;
        float res_25246 = fpow32(2.7182817F, exp_arg_25245);
        float x_25247 = 1.0F - res_25246;
        float B_25248 = x_25247 / a_24790;
        float x_25249 = B_25248 - x_25231;
        float x_25250 = y_24814 * x_25249;
        float A1_25251 = x_25250 / x_24810;
        float y_25252 = fpow32(B_25248, 2.0F);
        float x_25253 = x_24812 * y_25252;
        float A2_25254 = x_25253 / y_24815;
        float exp_arg_25255 = A1_25251 - A2_25254;
        float res_25256 = fpow32(2.7182817F, exp_arg_25255);
        float negate_arg_25257 = 5.0e-2F * B_25248;
        float exp_arg_25258 = 0.0F - negate_arg_25257;
        float res_25259 = fpow32(2.7182817F, exp_arg_25258);
        float res_25260 = res_25256 * res_25259;
        float res_25261 = res_25243 * res_25260;
        float res_25229 = res_25261 + redout_26337;
        
        ((float *) mem_26893)[i_26339] = res_25243;
        
        float redout_tmp_26974 = res_25229;
        
        redout_26337 = redout_tmp_26974;
    }
    res_25225 = redout_26337;
    
    float CVA_25264 = 6.0e-3F * res_25225;
    struct memblock mem_26907;
    
    mem_26907.references = NULL;
    if (memblock_alloc(ctx, &mem_26907, bytes_26620, "mem_26907")) {
        err = 1;
        goto cleanup;
    }
    memmove(mem_26907.mem + 0, mem_26893 + 0, steps_24786 *
            (int32_t) sizeof(float));
    out_arrsizze_26941 = steps_24786;
    if (memblock_set(ctx, &out_mem_26940, &mem_26907, "mem_26907") != 0)
        return 1;
    scalar_out_26939 = CVA_25264;
    *out_scalar_out_26978 = scalar_out_26939;
    (*out_mem_p_26979).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_26979, &out_mem_26940, "out_mem_26940") !=
        0)
        return 1;
    *out_out_arrsizze_26980 = out_arrsizze_26941;
    if (memblock_unref(ctx, &mem_26907, "mem_26907") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_26940, "out_mem_26940") != 0)
        return 1;
    
  cleanup:
    { }
    free(mem_26565);
    free(mem_26567);
    free(mem_26569);
    free(mem_26571);
    free(mem_26621);
    free(mem_26637);
    free(mem_26641);
    free(mem_26671);
    free(mem_26685);
    free(mem_26727);
    free(mem_26729);
    free(mem_26781);
    free(mem_26783);
    free(mem_26809);
    free(mem_26823);
    free(mem_26837);
    free(mem_26839);
    free(mem_26865);
    free(mem_26879);
    free(mem_26893);
    return err;
}
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_27001,
                       int64_t paths_25265, int64_t steps_25266)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_26562_cached_sizze_27002 = 0;
    char *mem_26562 = NULL;
    size_t mem_26564_cached_sizze_27003 = 0;
    char *mem_26564 = NULL;
    size_t mem_26566_cached_sizze_27004 = 0;
    char *mem_26566 = NULL;
    size_t mem_26568_cached_sizze_27005 = 0;
    char *mem_26568 = NULL;
    size_t mem_26570_cached_sizze_27006 = 0;
    char *mem_26570 = NULL;
    size_t mem_26572_cached_sizze_27007 = 0;
    char *mem_26572 = NULL;
    size_t mem_26574_cached_sizze_27008 = 0;
    char *mem_26574 = NULL;
    size_t mem_26632_cached_sizze_27009 = 0;
    char *mem_26632 = NULL;
    size_t mem_26648_cached_sizze_27010 = 0;
    char *mem_26648 = NULL;
    size_t mem_26652_cached_sizze_27011 = 0;
    char *mem_26652 = NULL;
    size_t mem_26688_cached_sizze_27012 = 0;
    char *mem_26688 = NULL;
    size_t mem_26702_cached_sizze_27013 = 0;
    char *mem_26702 = NULL;
    size_t mem_26744_cached_sizze_27014 = 0;
    char *mem_26744 = NULL;
    size_t mem_26746_cached_sizze_27015 = 0;
    char *mem_26746 = NULL;
    size_t mem_26798_cached_sizze_27016 = 0;
    char *mem_26798 = NULL;
    size_t mem_26800_cached_sizze_27017 = 0;
    char *mem_26800 = NULL;
    size_t mem_26826_cached_sizze_27018 = 0;
    char *mem_26826 = NULL;
    size_t mem_26840_cached_sizze_27019 = 0;
    char *mem_26840 = NULL;
    size_t mem_26854_cached_sizze_27020 = 0;
    char *mem_26854 = NULL;
    size_t mem_26856_cached_sizze_27021 = 0;
    char *mem_26856 = NULL;
    size_t mem_26882_cached_sizze_27022 = 0;
    char *mem_26882 = NULL;
    size_t mem_26896_cached_sizze_27023 = 0;
    char *mem_26896 = NULL;
    float scalar_out_26939;
    
    if (mem_26562_cached_sizze_27002 < (size_t) 180) {
        mem_26562 = realloc(mem_26562, 180);
        mem_26562_cached_sizze_27002 = 180;
    }
    
    struct memblock testzistatic_array_26940 = ctx->testzistatic_array_26940;
    
    memmove(mem_26562 + 0, testzistatic_array_26940.mem + 0, 45 *
            (int32_t) sizeof(float));
    if (mem_26564_cached_sizze_27003 < (size_t) 360) {
        mem_26564 = realloc(mem_26564, 360);
        mem_26564_cached_sizze_27003 = 360;
    }
    
    struct memblock testzistatic_array_26941 = ctx->testzistatic_array_26941;
    
    memmove(mem_26564 + 0, testzistatic_array_26941.mem + 0, 45 *
            (int32_t) sizeof(int64_t));
    if (mem_26566_cached_sizze_27004 < (size_t) 180) {
        mem_26566 = realloc(mem_26566, 180);
        mem_26566_cached_sizze_27004 = 180;
    }
    
    struct memblock testzistatic_array_26942 = ctx->testzistatic_array_26942;
    
    memmove(mem_26566 + 0, testzistatic_array_26942.mem + 0, 45 *
            (int32_t) sizeof(float));
    
    float res_25270;
    float redout_26201 = -INFINITY;
    
    for (int32_t i_26347 = 0; i_26347 < 45; i_26347++) {
        int64_t i_26202 = sext_i32_i64(i_26347);
        float x_25274 = ((float *) mem_26566)[i_26202];
        int64_t x_25275 = ((int64_t *) mem_26564)[i_26202];
        float res_25276 = sitofp_i64_f32(x_25275);
        float res_25277 = x_25274 * res_25276;
        float res_25273 = fmax32(res_25277, redout_26201);
        float redout_tmp_26943 = res_25273;
        
        redout_26201 = redout_tmp_26943;
    }
    res_25270 = redout_26201;
    
    float res_25278 = sitofp_i64_f32(steps_25266);
    float dt_25279 = res_25270 / res_25278;
    
    if (mem_26568_cached_sizze_27005 < (size_t) 180) {
        mem_26568 = realloc(mem_26568, 180);
        mem_26568_cached_sizze_27005 = 180;
    }
    if (mem_26570_cached_sizze_27006 < (size_t) 180) {
        mem_26570 = realloc(mem_26570, 180);
        mem_26570_cached_sizze_27006 = 180;
    }
    if (mem_26572_cached_sizze_27007 < (size_t) 360) {
        mem_26572 = realloc(mem_26572, 360);
        mem_26572_cached_sizze_27007 = 360;
    }
    if (mem_26574_cached_sizze_27008 < (size_t) 180) {
        mem_26574 = realloc(mem_26574, 180);
        mem_26574_cached_sizze_27008 = 180;
    }
    for (int32_t i_26355 = 0; i_26355 < 45; i_26355++) {
        int64_t i_26213 = sext_i32_i64(i_26355);
        bool x_25286 = sle64(0, i_26213);
        bool y_25287 = slt64(i_26213, 45);
        bool bounds_check_25288 = x_25286 && y_25287;
        bool index_certs_25289;
        
        if (!bounds_check_25288) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", i_26213,
                                   "] out of bounds for array of shape [", 45,
                                   "].",
                                   "-> #0  cva3d.fut:107:15-26\n   #1  cva3d.fut:106:17-110:85\n   #2  cva3d.fut:155:3-157:129\n   #3  cva3d.fut:154:1-157:137\n");
            return 1;
        }
        
        float res_25290 = ((float *) mem_26566)[i_26213];
        int64_t res_25291 = ((int64_t *) mem_26564)[i_26213];
        bool bounds_invalid_upwards_25293 = slt64(res_25291, 1);
        bool valid_25294 = !bounds_invalid_upwards_25293;
        bool range_valid_c_25295;
        
        if (!valid_25294) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_25291,
                          " is invalid.",
                          "-> #0  cva3d.fut:55:29-48\n   #1  cva3d.fut:96:25-65\n   #2  cva3d.fut:110:16-62\n   #3  cva3d.fut:106:17-110:85\n   #4  cva3d.fut:155:3-157:129\n   #5  cva3d.fut:154:1-157:137\n");
            return 1;
        }
        
        bool y_25297 = slt64(0, res_25291);
        bool index_certs_25298;
        
        if (!y_25297) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_25291, "].",
                                   "-> #0  cva3d.fut:97:47-70\n   #1  cva3d.fut:110:16-62\n   #2  cva3d.fut:106:17-110:85\n   #3  cva3d.fut:155:3-157:129\n   #4  cva3d.fut:154:1-157:137\n");
            return 1;
        }
        
        float binop_y_25299 = sitofp_i64_f32(res_25291);
        float index_primexp_25300 = res_25290 * binop_y_25299;
        float negate_arg_25301 = 1.0e-2F * index_primexp_25300;
        float exp_arg_25302 = 0.0F - negate_arg_25301;
        float res_25303 = fpow32(2.7182817F, exp_arg_25302);
        float x_25304 = 1.0F - res_25303;
        float B_25305 = x_25304 / 1.0e-2F;
        float x_25306 = B_25305 - index_primexp_25300;
        float x_25307 = 4.4999997e-6F * x_25306;
        float A1_25308 = x_25307 / 1.0e-4F;
        float y_25309 = fpow32(B_25305, 2.0F);
        float x_25310 = 1.0000001e-6F * y_25309;
        float A2_25311 = x_25310 / 4.0e-2F;
        float exp_arg_25312 = A1_25308 - A2_25311;
        float res_25313 = fpow32(2.7182817F, exp_arg_25312);
        float negate_arg_25314 = 5.0e-2F * B_25305;
        float exp_arg_25315 = 0.0F - negate_arg_25314;
        float res_25316 = fpow32(2.7182817F, exp_arg_25315);
        float res_25317 = res_25313 * res_25316;
        float res_25318;
        float redout_26203 = 0.0F;
        
        for (int64_t i_26204 = 0; i_26204 < res_25291; i_26204++) {
            int64_t index_primexp_26349 = add64(1, i_26204);
            float res_25323 = sitofp_i64_f32(index_primexp_26349);
            float res_25324 = res_25290 * res_25323;
            float negate_arg_25325 = 1.0e-2F * res_25324;
            float exp_arg_25326 = 0.0F - negate_arg_25325;
            float res_25327 = fpow32(2.7182817F, exp_arg_25326);
            float x_25328 = 1.0F - res_25327;
            float B_25329 = x_25328 / 1.0e-2F;
            float x_25330 = B_25329 - res_25324;
            float x_25331 = 4.4999997e-6F * x_25330;
            float A1_25332 = x_25331 / 1.0e-4F;
            float y_25333 = fpow32(B_25329, 2.0F);
            float x_25334 = 1.0000001e-6F * y_25333;
            float A2_25335 = x_25334 / 4.0e-2F;
            float exp_arg_25336 = A1_25332 - A2_25335;
            float res_25337 = fpow32(2.7182817F, exp_arg_25336);
            float negate_arg_25338 = 5.0e-2F * B_25329;
            float exp_arg_25339 = 0.0F - negate_arg_25338;
            float res_25340 = fpow32(2.7182817F, exp_arg_25339);
            float res_25341 = res_25337 * res_25340;
            float res_25321 = res_25341 + redout_26203;
            float redout_tmp_26948 = res_25321;
            
            redout_26203 = redout_tmp_26948;
        }
        res_25318 = redout_26203;
        
        float x_25342 = 1.0F - res_25317;
        float y_25343 = res_25290 * res_25318;
        float res_25344 = x_25342 / y_25343;
        
        ((float *) mem_26568)[i_26213] = res_25344;
        memmove(mem_26570 + i_26213 * 4, mem_26562 + i_26213 * 4,
                (int32_t) sizeof(float));
        memmove(mem_26572 + i_26213 * 8, mem_26564 + i_26213 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_26574 + i_26213 * 4, mem_26566 + i_26213 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_25345 = res_25278 / res_25270;
    bool bounds_invalid_upwards_25346 = slt64(steps_25266, 1);
    bool valid_25347 = !bounds_invalid_upwards_25346;
    bool range_valid_c_25348;
    
    if (!valid_25347) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_25266,
                               " is invalid.",
                               "-> #0  cva3d.fut:61:56-67\n   #1  cva3d.fut:112:17-44\n   #2  cva3d.fut:155:3-157:129\n   #3  cva3d.fut:154:1-157:137\n");
        return 1;
    }
    
    int64_t bytes_26631 = 4 * steps_25266;
    
    if (mem_26632_cached_sizze_27009 < (size_t) bytes_26631) {
        mem_26632 = realloc(mem_26632, bytes_26631);
        mem_26632_cached_sizze_27009 = bytes_26631;
    }
    for (int64_t i_26220 = 0; i_26220 < steps_25266; i_26220++) {
        int64_t index_primexp_26357 = add64(1, i_26220);
        float res_25352 = sitofp_i64_f32(index_primexp_26357);
        float res_25353 = res_25352 / sims_per_year_25345;
        
        ((float *) mem_26632)[i_26220] = res_25353;
    }
    
    bool bounds_invalid_upwards_25354 = slt64(paths_25265, 0);
    bool valid_25355 = !bounds_invalid_upwards_25354;
    bool range_valid_c_25356;
    
    if (!valid_25355) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_25265,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva3d.fut:116:19-49\n   #4  cva3d.fut:155:3-157:129\n   #5  cva3d.fut:154:1-157:137\n");
        return 1;
    }
    
    int64_t upper_bound_25359 = sub64(steps_25266, 1);
    float res_25360;
    
    res_25360 = futrts_sqrt32(dt_25279);
    
    int64_t binop_x_26646 = paths_25265 * steps_25266;
    int64_t binop_x_26647 = 45 * binop_x_26646;
    int64_t bytes_26645 = 4 * binop_x_26647;
    
    if (mem_26648_cached_sizze_27010 < (size_t) bytes_26645) {
        mem_26648 = realloc(mem_26648, bytes_26645);
        mem_26648_cached_sizze_27010 = bytes_26645;
    }
    if (mem_26652_cached_sizze_27011 < (size_t) bytes_26645) {
        mem_26652 = realloc(mem_26652, bytes_26645);
        mem_26652_cached_sizze_27011 = bytes_26645;
    }
    
    int64_t ctx_val_26677 = 45 * steps_25266;
    
    if (mem_26688_cached_sizze_27012 < (size_t) bytes_26631) {
        mem_26688 = realloc(mem_26688, bytes_26631);
        mem_26688_cached_sizze_27012 = bytes_26631;
    }
    if (mem_26702_cached_sizze_27013 < (size_t) bytes_26631) {
        mem_26702 = realloc(mem_26702, bytes_26631);
        mem_26702_cached_sizze_27013 = bytes_26631;
    }
    if (mem_26744_cached_sizze_27014 < (size_t) 180) {
        mem_26744 = realloc(mem_26744, 180);
        mem_26744_cached_sizze_27014 = 180;
    }
    if (mem_26746_cached_sizze_27015 < (size_t) 180) {
        mem_26746 = realloc(mem_26746, 180);
        mem_26746_cached_sizze_27015 = 180;
    }
    for (int64_t i_26237 = 0; i_26237 < paths_25265; i_26237++) {
        int32_t res_25364 = sext_i64_i32(i_26237);
        int32_t x_25365 = lshr32(res_25364, 16);
        int32_t x_25366 = res_25364 ^ x_25365;
        int32_t x_25367 = mul32(73244475, x_25366);
        int32_t x_25368 = lshr32(x_25367, 16);
        int32_t x_25369 = x_25367 ^ x_25368;
        int32_t x_25370 = mul32(73244475, x_25369);
        int32_t x_25371 = lshr32(x_25370, 16);
        int32_t x_25372 = x_25370 ^ x_25371;
        int32_t unsign_arg_25373 = 777822902 ^ x_25372;
        int32_t unsign_arg_25374 = mul32(48271, unsign_arg_25373);
        int32_t unsign_arg_25375 = umod32(unsign_arg_25374, 2147483647);
        
        for (int64_t i_26224 = 0; i_26224 < steps_25266; i_26224++) {
            int32_t res_25378 = sext_i64_i32(i_26224);
            int32_t x_25379 = lshr32(res_25378, 16);
            int32_t x_25380 = res_25378 ^ x_25379;
            int32_t x_25381 = mul32(73244475, x_25380);
            int32_t x_25382 = lshr32(x_25381, 16);
            int32_t x_25383 = x_25381 ^ x_25382;
            int32_t x_25384 = mul32(73244475, x_25383);
            int32_t x_25385 = lshr32(x_25384, 16);
            int32_t x_25386 = x_25384 ^ x_25385;
            int32_t unsign_arg_25387 = unsign_arg_25375 ^ x_25386;
            int32_t unsign_arg_25388 = mul32(48271, unsign_arg_25387);
            int32_t unsign_arg_25389 = umod32(unsign_arg_25388, 2147483647);
            int32_t unsign_arg_25390 = mul32(48271, unsign_arg_25389);
            int32_t unsign_arg_25391 = umod32(unsign_arg_25390, 2147483647);
            float res_25392 = uitofp_i32_f32(unsign_arg_25389);
            float res_25393 = res_25392 / 2.1474836e9F;
            float res_25394 = uitofp_i32_f32(unsign_arg_25391);
            float res_25395 = res_25394 / 2.1474836e9F;
            float res_25396;
            
            res_25396 = futrts_log32(res_25393);
            
            float res_25397 = -2.0F * res_25396;
            float res_25398;
            
            res_25398 = futrts_sqrt32(res_25397);
            
            float res_25399 = 6.2831855F * res_25395;
            float res_25400;
            
            res_25400 = futrts_cos32(res_25399);
            
            float res_25401 = res_25398 * res_25400;
            
            ((float *) mem_26688)[i_26224] = res_25401;
        }
        for (int64_t i_26953 = 0; i_26953 < steps_25266; i_26953++) {
            ((float *) mem_26702)[i_26953] = 5.0e-2F;
        }
        for (int64_t i_25404 = 0; i_25404 < upper_bound_25359; i_25404++) {
            bool y_25406 = slt64(i_25404, steps_25266);
            bool index_certs_25407;
            
            if (!y_25406) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25404,
                              "] out of bounds for array of shape [",
                              steps_25266, "].",
                              "-> #0  cva3d.fut:72:97-104\n   #1  cva3d.fut:124:32-62\n   #2  cva3d.fut:124:22-69\n   #3  cva3d.fut:155:3-157:129\n   #4  cva3d.fut:154:1-157:137\n");
                return 1;
            }
            
            float shortstep_arg_25408 = ((float *) mem_26688)[i_25404];
            float shortstep_arg_25409 = ((float *) mem_26702)[i_25404];
            float y_25410 = 5.0e-2F - shortstep_arg_25409;
            float x_25411 = 1.0e-2F * y_25410;
            float x_25412 = dt_25279 * x_25411;
            float x_25413 = res_25360 * shortstep_arg_25408;
            float y_25414 = 1.0e-3F * x_25413;
            float delta_r_25415 = x_25412 + y_25414;
            float res_25416 = shortstep_arg_25409 + delta_r_25415;
            int64_t i_25417 = add64(1, i_25404);
            bool x_25418 = sle64(0, i_25417);
            bool y_25419 = slt64(i_25417, steps_25266);
            bool bounds_check_25420 = x_25418 && y_25419;
            bool index_certs_25421;
            
            if (!bounds_check_25420) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25417,
                              "] out of bounds for array of shape [",
                              steps_25266, "].",
                              "-> #0  cva3d.fut:72:58-105\n   #1  cva3d.fut:124:32-62\n   #2  cva3d.fut:124:22-69\n   #3  cva3d.fut:155:3-157:129\n   #4  cva3d.fut:154:1-157:137\n");
                return 1;
            }
            ((float *) mem_26702)[i_25417] = res_25416;
        }
        for (int64_t i_26230 = 0; i_26230 < steps_25266; i_26230++) {
            float x_25425 = ((float *) mem_26702)[i_26230];
            float x_25426 = ((float *) mem_26632)[i_26230];
            
            for (int64_t i_26957 = 0; i_26957 < 45; i_26957++) {
                ((float *) mem_26744)[i_26957] = x_25425;
            }
            for (int64_t i_26958 = 0; i_26958 < 45; i_26958++) {
                ((float *) mem_26746)[i_26958] = x_25426;
            }
            memmove(mem_26648 + (i_26237 * ctx_val_26677 + i_26230 * 45) * 4,
                    mem_26744 + 0, 45 * (int32_t) sizeof(float));
            memmove(mem_26652 + (i_26237 * ctx_val_26677 + i_26230 * 45) * 4,
                    mem_26746 + 0, 45 * (int32_t) sizeof(float));
        }
    }
    
    int64_t flat_dim_25430 = 45 * binop_x_26646;
    int64_t bytes_26797 = 8 * flat_dim_25430;
    
    if (mem_26798_cached_sizze_27016 < (size_t) bytes_26797) {
        mem_26798 = realloc(mem_26798, bytes_26797);
        mem_26798_cached_sizze_27016 = bytes_26797;
    }
    if (mem_26800_cached_sizze_27017 < (size_t) bytes_26797) {
        mem_26800 = realloc(mem_26800, bytes_26797);
        mem_26800_cached_sizze_27017 = bytes_26797;
    }
    
    int64_t discard_26248;
    int64_t scanacc_26242 = 0;
    
    for (int64_t i_26245 = 0; i_26245 < flat_dim_25430; i_26245++) {
        int64_t binop_x_26371 = squot64(i_26245, ctx_val_26677);
        int64_t binop_y_26373 = binop_x_26371 * ctx_val_26677;
        int64_t binop_x_26374 = i_26245 - binop_y_26373;
        int64_t binop_x_26380 = squot64(binop_x_26374, 45);
        int64_t binop_y_26381 = 45 * binop_x_26380;
        int64_t new_index_26382 = binop_x_26374 - binop_y_26381;
        int64_t x_25441 = ((int64_t *) mem_26572)[new_index_26382];
        float x_25442 = ((float *) mem_26574)[new_index_26382];
        float x_25443 = ((float *) mem_26652)[binop_x_26371 * ctx_val_26677 +
                                              binop_x_26380 * 45 +
                                              new_index_26382];
        float x_25444 = x_25443 / x_25442;
        float ceil_arg_25445 = x_25444 - 1.0F;
        float res_25446;
        
        res_25446 = futrts_ceil32(ceil_arg_25445);
        
        int64_t res_25447 = fptosi_f32_i64(res_25446);
        int64_t max_arg_25448 = sub64(x_25441, res_25447);
        int64_t res_25449 = smax64(0, max_arg_25448);
        bool cond_25450 = res_25449 == 0;
        int64_t res_25451;
        
        if (cond_25450) {
            res_25451 = 1;
        } else {
            res_25451 = res_25449;
        }
        
        int64_t res_25440 = add64(res_25451, scanacc_26242);
        
        ((int64_t *) mem_26798)[i_26245] = res_25440;
        ((int64_t *) mem_26800)[i_26245] = res_25451;
        
        int64_t scanacc_tmp_26959 = res_25440;
        
        scanacc_26242 = scanacc_tmp_26959;
    }
    discard_26248 = scanacc_26242;
    
    int64_t res_25454;
    int64_t redout_26249 = 0;
    
    for (int64_t i_26250 = 0; i_26250 < flat_dim_25430; i_26250++) {
        int64_t x_25458 = ((int64_t *) mem_26800)[i_26250];
        int64_t res_25457 = add64(x_25458, redout_26249);
        int64_t redout_tmp_26962 = res_25457;
        
        redout_26249 = redout_tmp_26962;
    }
    res_25454 = redout_26249;
    
    bool bounds_invalid_upwards_25459 = slt64(res_25454, 0);
    bool valid_25460 = !bounds_invalid_upwards_25459;
    bool range_valid_c_25461;
    
    if (!valid_25460) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", res_25454,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:155:3-157:129\n   #7  cva3d.fut:154:1-157:137\n");
        return 1;
    }
    
    int64_t bytes_26825 = 8 * res_25454;
    
    if (mem_26826_cached_sizze_27018 < (size_t) bytes_26825) {
        mem_26826 = realloc(mem_26826, bytes_26825);
        mem_26826_cached_sizze_27018 = bytes_26825;
    }
    for (int64_t i_26963 = 0; i_26963 < res_25454; i_26963++) {
        ((int64_t *) mem_26826)[i_26963] = 0;
    }
    for (int64_t iter_26251 = 0; iter_26251 < flat_dim_25430; iter_26251++) {
        int64_t i_p_o_26427 = add64(-1, iter_26251);
        int64_t rot_i_26428 = smod64(i_p_o_26427, flat_dim_25430);
        int64_t pixel_26254 = ((int64_t *) mem_26798)[rot_i_26428];
        bool cond_25469 = iter_26251 == 0;
        int64_t res_25470;
        
        if (cond_25469) {
            res_25470 = 0;
        } else {
            res_25470 = pixel_26254;
        }
        
        bool less_than_zzero_26255 = slt64(res_25470, 0);
        bool greater_than_sizze_26256 = sle64(res_25454, res_25470);
        bool outside_bounds_dim_26257 = less_than_zzero_26255 ||
             greater_than_sizze_26256;
        
        if (!outside_bounds_dim_26257) {
            int64_t read_hist_26259 = ((int64_t *) mem_26826)[res_25470];
            int64_t res_25466 = smax64(iter_26251, read_hist_26259);
            
            ((int64_t *) mem_26826)[res_25470] = res_25466;
        }
    }
    if (mem_26840_cached_sizze_27019 < (size_t) bytes_26825) {
        mem_26840 = realloc(mem_26840, bytes_26825);
        mem_26840_cached_sizze_27019 = bytes_26825;
    }
    
    int64_t discard_26272;
    int64_t scanacc_26265 = 0;
    
    for (int64_t i_26268 = 0; i_26268 < res_25454; i_26268++) {
        int64_t x_25480 = ((int64_t *) mem_26826)[i_26268];
        bool res_25481 = slt64(0, x_25480);
        int64_t res_25478;
        
        if (res_25481) {
            res_25478 = x_25480;
        } else {
            int64_t res_25479 = add64(x_25480, scanacc_26265);
            
            res_25478 = res_25479;
        }
        ((int64_t *) mem_26840)[i_26268] = res_25478;
        
        int64_t scanacc_tmp_26965 = res_25478;
        
        scanacc_26265 = scanacc_tmp_26965;
    }
    discard_26272 = scanacc_26265;
    
    int64_t bytes_26853 = 4 * res_25454;
    
    if (mem_26854_cached_sizze_27020 < (size_t) bytes_26853) {
        mem_26854 = realloc(mem_26854, bytes_26853);
        mem_26854_cached_sizze_27020 = bytes_26853;
    }
    if (mem_26856_cached_sizze_27021 < (size_t) res_25454) {
        mem_26856 = realloc(mem_26856, res_25454);
        mem_26856_cached_sizze_27021 = res_25454;
    }
    
    int64_t lstel_tmp_25533 = 1;
    int64_t inpacc_25491;
    float inpacc_25493;
    int64_t inpacc_25499;
    float inpacc_25501;
    
    inpacc_25499 = 0;
    inpacc_25501 = 0.0F;
    for (int64_t i_26311 = 0; i_26311 < res_25454; i_26311++) {
        int64_t x_26442 = ((int64_t *) mem_26840)[i_26311];
        int64_t i_p_o_26444 = add64(-1, i_26311);
        int64_t rot_i_26445 = smod64(i_p_o_26444, res_25454);
        int64_t x_26446 = ((int64_t *) mem_26840)[rot_i_26445];
        bool res_26447 = x_26442 == x_26446;
        bool res_26448 = !res_26447;
        int64_t res_25535;
        
        if (res_26448) {
            res_25535 = lstel_tmp_25533;
        } else {
            int64_t res_25536 = add64(1, inpacc_25499);
            
            res_25535 = res_25536;
        }
        
        int64_t res_25550;
        
        if (res_26448) {
            res_25550 = 1;
        } else {
            int64_t res_25551 = add64(1, inpacc_25499);
            
            res_25550 = res_25551;
        }
        
        int64_t res_25552 = sub64(res_25550, 1);
        bool x_26463 = sle64(0, x_26442);
        bool y_26464 = slt64(x_26442, flat_dim_25430);
        bool bounds_check_26465 = x_26463 && y_26464;
        bool index_certs_26466;
        
        if (!bounds_check_26465) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", x_26442,
                                   "] out of bounds for array of shape [",
                                   flat_dim_25430, "].",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:155:3-157:129\n   #7  cva3d.fut:154:1-157:137\n");
            return 1;
        }
        
        int64_t new_index_26467 = squot64(x_26442, ctx_val_26677);
        int64_t binop_y_26468 = new_index_26467 * ctx_val_26677;
        int64_t binop_x_26469 = x_26442 - binop_y_26468;
        int64_t new_index_26470 = squot64(binop_x_26469, 45);
        int64_t binop_y_26471 = 45 * new_index_26470;
        int64_t new_index_26472 = binop_x_26469 - binop_y_26471;
        float lifted_0_get_arg_26473 = ((float *) mem_26648)[new_index_26467 *
                                                             ctx_val_26677 +
                                                             new_index_26470 *
                                                             45 +
                                                             new_index_26472];
        float lifted_0_get_arg_26474 = ((float *) mem_26568)[new_index_26472];
        float lifted_0_get_arg_26475 = ((float *) mem_26570)[new_index_26472];
        int64_t lifted_0_get_arg_26476 =
                ((int64_t *) mem_26572)[new_index_26472];
        float lifted_0_get_arg_26477 = ((float *) mem_26574)[new_index_26472];
        float lifted_0_get_arg_26478 = ((float *) mem_26652)[new_index_26467 *
                                                             ctx_val_26677 +
                                                             new_index_26470 *
                                                             45 +
                                                             new_index_26472];
        float x_26479 = lifted_0_get_arg_26478 / lifted_0_get_arg_26477;
        float ceil_arg_26480 = x_26479 - 1.0F;
        float res_26481;
        
        res_26481 = futrts_ceil32(ceil_arg_26480);
        
        int64_t res_26482 = fptosi_f32_i64(res_26481);
        int64_t max_arg_26483 = sub64(lifted_0_get_arg_26476, res_26482);
        int64_t res_26484 = smax64(0, max_arg_26483);
        bool cond_26485 = res_26484 == 0;
        float res_26486;
        
        if (cond_26485) {
            res_26486 = 0.0F;
        } else {
            float res_26487;
            
            res_26487 = futrts_ceil32(x_26479);
            
            float start_26488 = lifted_0_get_arg_26477 * res_26487;
            float res_26489;
            
            res_26489 = futrts_ceil32(ceil_arg_26480);
            
            int64_t res_26490 = fptosi_f32_i64(res_26489);
            int64_t max_arg_26491 = sub64(lifted_0_get_arg_26476, res_26490);
            int64_t res_26492 = smax64(0, max_arg_26491);
            int64_t sizze_26493 = sub64(res_26492, 1);
            bool cond_26494 = res_25552 == 0;
            float res_26495;
            
            if (cond_26494) {
                res_26495 = 1.0F;
            } else {
                res_26495 = 0.0F;
            }
            
            bool cond_26496 = slt64(0, res_25552);
            float res_26497;
            
            if (cond_26496) {
                float y_26498 = lifted_0_get_arg_26474 * lifted_0_get_arg_26477;
                float res_26499 = res_26495 - y_26498;
                
                res_26497 = res_26499;
            } else {
                res_26497 = res_26495;
            }
            
            bool cond_26500 = res_25552 == sizze_26493;
            float res_26501;
            
            if (cond_26500) {
                float res_26502 = res_26497 - 1.0F;
                
                res_26501 = res_26502;
            } else {
                res_26501 = res_26497;
            }
            
            float res_26503 = lifted_0_get_arg_26475 * res_26501;
            float res_26504 = sitofp_i64_f32(res_25552);
            float y_26505 = lifted_0_get_arg_26477 * res_26504;
            float bondprice_arg_26506 = start_26488 + y_26505;
            float y_26507 = bondprice_arg_26506 - lifted_0_get_arg_26478;
            float negate_arg_26508 = 1.0e-2F * y_26507;
            float exp_arg_26509 = 0.0F - negate_arg_26508;
            float res_26510 = fpow32(2.7182817F, exp_arg_26509);
            float x_26511 = 1.0F - res_26510;
            float B_26512 = x_26511 / 1.0e-2F;
            float x_26513 = B_26512 - bondprice_arg_26506;
            float x_26514 = lifted_0_get_arg_26478 + x_26513;
            float x_26515 = 4.4999997e-6F * x_26514;
            float A1_26516 = x_26515 / 1.0e-4F;
            float y_26517 = fpow32(B_26512, 2.0F);
            float x_26518 = 1.0000001e-6F * y_26517;
            float A2_26519 = x_26518 / 4.0e-2F;
            float exp_arg_26520 = A1_26516 - A2_26519;
            float res_26521 = fpow32(2.7182817F, exp_arg_26520);
            float negate_arg_26522 = lifted_0_get_arg_26473 * B_26512;
            float exp_arg_26523 = 0.0F - negate_arg_26522;
            float res_26524 = fpow32(2.7182817F, exp_arg_26523);
            float res_26525 = res_26521 * res_26524;
            float res_26526 = res_26503 * res_26525;
            
            res_26486 = res_26526;
        }
        
        float lstel_tmp_25620 = res_26486;
        float res_25626;
        
        if (res_26448) {
            res_25626 = res_26486;
        } else {
            float res_25627 = inpacc_25501 + res_26486;
            
            res_25626 = res_25627;
        }
        
        float res_25629;
        
        if (res_26448) {
            res_25629 = lstel_tmp_25620;
        } else {
            float res_25630 = inpacc_25501 + res_26486;
            
            res_25629 = res_25630;
        }
        ((float *) mem_26854)[i_26311] = res_25626;
        ((bool *) mem_26856)[i_26311] = res_26448;
        
        int64_t inpacc_tmp_26967 = res_25535;
        float inpacc_tmp_26968 = res_25629;
        
        inpacc_25499 = inpacc_tmp_26967;
        inpacc_25501 = inpacc_tmp_26968;
    }
    inpacc_25491 = inpacc_25499;
    inpacc_25493 = inpacc_25501;
    if (mem_26882_cached_sizze_27022 < (size_t) bytes_26825) {
        mem_26882 = realloc(mem_26882, bytes_26825);
        mem_26882_cached_sizze_27022 = bytes_26825;
    }
    
    int64_t discard_26320;
    int64_t scanacc_26316 = 0;
    
    for (int64_t i_26318 = 0; i_26318 < res_25454; i_26318++) {
        int64_t i_p_o_26537 = add64(1, i_26318);
        int64_t rot_i_26538 = smod64(i_p_o_26537, res_25454);
        bool x_25643 = ((bool *) mem_26856)[rot_i_26538];
        int64_t res_25644 = btoi_bool_i64(x_25643);
        int64_t res_25642 = add64(res_25644, scanacc_26316);
        
        ((int64_t *) mem_26882)[i_26318] = res_25642;
        
        int64_t scanacc_tmp_26971 = res_25642;
        
        scanacc_26316 = scanacc_tmp_26971;
    }
    discard_26320 = scanacc_26316;
    
    bool cond_25645 = slt64(0, res_25454);
    int64_t num_segments_25646;
    
    if (cond_25645) {
        int64_t i_25647 = sub64(res_25454, 1);
        bool x_25648 = sle64(0, i_25647);
        bool y_25649 = slt64(i_25647, res_25454);
        bool bounds_check_25650 = x_25648 && y_25649;
        bool index_certs_25651;
        
        if (!bounds_check_25650) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", i_25647,
                                   "] out of bounds for array of shape [",
                                   res_25454, "].",
                                   "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva3d.fut:135:18-77\n   #5  cva3d.fut:155:3-157:129\n   #6  cva3d.fut:154:1-157:137\n");
            return 1;
        }
        
        int64_t res_25652 = ((int64_t *) mem_26882)[i_25647];
        
        num_segments_25646 = res_25652;
    } else {
        num_segments_25646 = 0;
    }
    
    bool bounds_invalid_upwards_25653 = slt64(num_segments_25646, 0);
    bool valid_25654 = !bounds_invalid_upwards_25653;
    bool range_valid_c_25655;
    
    if (!valid_25654) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", num_segments_25646,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:155:3-157:129\n   #7  cva3d.fut:154:1-157:137\n");
        return 1;
    }
    
    int64_t bytes_26895 = 4 * num_segments_25646;
    
    if (mem_26896_cached_sizze_27023 < (size_t) bytes_26895) {
        mem_26896 = realloc(mem_26896, bytes_26895);
        mem_26896_cached_sizze_27023 = bytes_26895;
    }
    for (int64_t i_26973 = 0; i_26973 < num_segments_25646; i_26973++) {
        ((float *) mem_26896)[i_26973] = 0.0F;
    }
    for (int64_t write_iter_26321 = 0; write_iter_26321 < res_25454;
         write_iter_26321++) {
        int64_t write_iv_26323 = ((int64_t *) mem_26882)[write_iter_26321];
        int64_t i_p_o_26540 = add64(1, write_iter_26321);
        int64_t rot_i_26541 = smod64(i_p_o_26540, res_25454);
        bool write_iv_26324 = ((bool *) mem_26856)[rot_i_26541];
        int64_t res_25661;
        
        if (write_iv_26324) {
            int64_t res_25662 = sub64(write_iv_26323, 1);
            
            res_25661 = res_25662;
        } else {
            res_25661 = -1;
        }
        
        bool less_than_zzero_26326 = slt64(res_25661, 0);
        bool greater_than_sizze_26327 = sle64(num_segments_25646, res_25661);
        bool outside_bounds_dim_26328 = less_than_zzero_26326 ||
             greater_than_sizze_26327;
        
        if (!outside_bounds_dim_26328) {
            memmove(mem_26896 + res_25661 * 4, mem_26854 + write_iter_26321 * 4,
                    (int32_t) sizeof(float));
        }
    }
    
    bool dim_match_25663 = flat_dim_25430 == num_segments_25646;
    bool empty_or_match_cert_25664;
    
    if (!dim_match_25663) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Value of (core language) shape (",
                               num_segments_25646,
                               ") cannot match shape of type `[",
                               flat_dim_25430, "]b`.",
                               "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva3d.fut:135:18-77\n   #2  cva3d.fut:155:3-157:129\n   #3  cva3d.fut:154:1-157:137\n");
        return 1;
    }
    
    float res_25665 = sitofp_i64_f32(paths_25265);
    float res_25668;
    float redout_26336 = 0.0F;
    
    for (int64_t i_26337 = 0; i_26337 < steps_25266; i_26337++) {
        float x_25673 = ((float *) mem_26632)[i_26337];
        int64_t binop_y_26545 = 45 * i_26337;
        float res_25674;
        float redout_26334 = 0.0F;
        
        for (int64_t i_26335 = 0; i_26335 < paths_25265; i_26335++) {
            int64_t binop_x_26544 = i_26335 * ctx_val_26677;
            int64_t binop_x_26546 = binop_x_26544 + binop_y_26545;
            float res_25679;
            float redout_26332 = 0.0F;
            
            for (int32_t i_26542 = 0; i_26542 < 45; i_26542++) {
                int64_t i_26333 = sext_i32_i64(i_26542);
                int64_t new_index_26547 = i_26333 + binop_x_26546;
                float x_25683 = ((float *) mem_26896)[new_index_26547];
                float res_25682 = x_25683 + redout_26332;
                float redout_tmp_26977 = res_25682;
                
                redout_26332 = redout_tmp_26977;
            }
            res_25679 = redout_26332;
            
            float res_25684 = fmax32(0.0F, res_25679);
            float res_25677 = res_25684 + redout_26334;
            float redout_tmp_26976 = res_25677;
            
            redout_26334 = redout_tmp_26976;
        }
        res_25674 = redout_26334;
        
        float res_25685 = res_25674 / res_25665;
        float negate_arg_25686 = 1.0e-2F * x_25673;
        float exp_arg_25687 = 0.0F - negate_arg_25686;
        float res_25688 = fpow32(2.7182817F, exp_arg_25687);
        float x_25689 = 1.0F - res_25688;
        float B_25690 = x_25689 / 1.0e-2F;
        float x_25691 = B_25690 - x_25673;
        float x_25692 = 4.4999997e-6F * x_25691;
        float A1_25693 = x_25692 / 1.0e-4F;
        float y_25694 = fpow32(B_25690, 2.0F);
        float x_25695 = 1.0000001e-6F * y_25694;
        float A2_25696 = x_25695 / 4.0e-2F;
        float exp_arg_25697 = A1_25693 - A2_25696;
        float res_25698 = fpow32(2.7182817F, exp_arg_25697);
        float negate_arg_25699 = 5.0e-2F * B_25690;
        float exp_arg_25700 = 0.0F - negate_arg_25699;
        float res_25701 = fpow32(2.7182817F, exp_arg_25700);
        float res_25702 = res_25698 * res_25701;
        float res_25703 = res_25685 * res_25702;
        float res_25671 = res_25703 + redout_26336;
        float redout_tmp_26975 = res_25671;
        
        redout_26336 = redout_tmp_26975;
    }
    res_25668 = redout_26336;
    
    float CVA_25704 = 6.0e-3F * res_25668;
    
    scalar_out_26939 = CVA_25704;
    *out_scalar_out_27001 = scalar_out_26939;
    
  cleanup:
    { }
    free(mem_26562);
    free(mem_26564);
    free(mem_26566);
    free(mem_26568);
    free(mem_26570);
    free(mem_26572);
    free(mem_26574);
    free(mem_26632);
    free(mem_26648);
    free(mem_26652);
    free(mem_26688);
    free(mem_26702);
    free(mem_26744);
    free(mem_26746);
    free(mem_26798);
    free(mem_26800);
    free(mem_26826);
    free(mem_26840);
    free(mem_26854);
    free(mem_26856);
    free(mem_26882);
    free(mem_26896);
    return err;
}
static int futrts_test2(struct futhark_context *ctx,
                        float *out_scalar_out_27027, int64_t paths_25705,
                        int64_t steps_25706, int64_t numswaps_25707)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_26562_cached_sizze_27028 = 0;
    char *mem_26562 = NULL;
    size_t mem_26564_cached_sizze_27029 = 0;
    char *mem_26564 = NULL;
    size_t mem_26566_cached_sizze_27030 = 0;
    char *mem_26566 = NULL;
    size_t mem_26604_cached_sizze_27031 = 0;
    char *mem_26604 = NULL;
    size_t mem_26606_cached_sizze_27032 = 0;
    char *mem_26606 = NULL;
    size_t mem_26608_cached_sizze_27033 = 0;
    char *mem_26608 = NULL;
    size_t mem_26610_cached_sizze_27034 = 0;
    char *mem_26610 = NULL;
    size_t mem_26660_cached_sizze_27035 = 0;
    char *mem_26660 = NULL;
    size_t mem_26676_cached_sizze_27036 = 0;
    char *mem_26676 = NULL;
    size_t mem_26680_cached_sizze_27037 = 0;
    char *mem_26680 = NULL;
    size_t mem_26710_cached_sizze_27038 = 0;
    char *mem_26710 = NULL;
    size_t mem_26724_cached_sizze_27039 = 0;
    char *mem_26724 = NULL;
    size_t mem_26766_cached_sizze_27040 = 0;
    char *mem_26766 = NULL;
    size_t mem_26768_cached_sizze_27041 = 0;
    char *mem_26768 = NULL;
    size_t mem_26820_cached_sizze_27042 = 0;
    char *mem_26820 = NULL;
    size_t mem_26822_cached_sizze_27043 = 0;
    char *mem_26822 = NULL;
    size_t mem_26848_cached_sizze_27044 = 0;
    char *mem_26848 = NULL;
    size_t mem_26862_cached_sizze_27045 = 0;
    char *mem_26862 = NULL;
    size_t mem_26876_cached_sizze_27046 = 0;
    char *mem_26876 = NULL;
    size_t mem_26878_cached_sizze_27047 = 0;
    char *mem_26878 = NULL;
    size_t mem_26904_cached_sizze_27048 = 0;
    char *mem_26904 = NULL;
    size_t mem_26918_cached_sizze_27049 = 0;
    char *mem_26918 = NULL;
    float scalar_out_26939;
    bool bounds_invalid_upwards_25708 = slt64(numswaps_25707, 0);
    bool valid_25709 = !bounds_invalid_upwards_25708;
    bool range_valid_c_25710;
    
    if (!valid_25709) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", numswaps_25707,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva3d.fut:172:29-62\n   #4  cva3d.fut:172:19-93\n   #5  cva3d.fut:170:1-176:75\n");
        return 1;
    }
    
    float res_25724 = sitofp_i64_f32(numswaps_25707);
    float res_25725 = res_25724 - 1.0F;
    int64_t bytes_26561 = 4 * numswaps_25707;
    
    if (mem_26562_cached_sizze_27028 < (size_t) bytes_26561) {
        mem_26562 = realloc(mem_26562, bytes_26561);
        mem_26562_cached_sizze_27028 = bytes_26561;
    }
    
    int64_t bytes_26563 = 8 * numswaps_25707;
    
    if (mem_26564_cached_sizze_27029 < (size_t) bytes_26563) {
        mem_26564 = realloc(mem_26564, bytes_26563);
        mem_26564_cached_sizze_27029 = bytes_26563;
    }
    if (mem_26566_cached_sizze_27030 < (size_t) bytes_26561) {
        mem_26566 = realloc(mem_26566, bytes_26561);
        mem_26566_cached_sizze_27030 = bytes_26561;
    }
    
    int64_t ctx_val_26583 = 1;
    float res_25737;
    float redout_26204 = -INFINITY;
    
    for (int64_t i_26208 = 0; i_26208 < numswaps_25707; i_26208++) {
        int32_t res_25745 = sext_i64_i32(i_26208);
        int32_t x_25746 = lshr32(res_25745, 16);
        int32_t x_25747 = res_25745 ^ x_25746;
        int32_t x_25748 = mul32(73244475, x_25747);
        int32_t x_25749 = lshr32(x_25748, 16);
        int32_t x_25750 = x_25748 ^ x_25749;
        int32_t x_25751 = mul32(73244475, x_25750);
        int32_t x_25752 = lshr32(x_25751, 16);
        int32_t x_25753 = x_25751 ^ x_25752;
        int32_t unsign_arg_25754 = 281253711 ^ x_25753;
        int32_t unsign_arg_25755 = mul32(48271, unsign_arg_25754);
        int32_t unsign_arg_25756 = umod32(unsign_arg_25755, 2147483647);
        float res_25757 = uitofp_i32_f32(unsign_arg_25756);
        float res_25758 = res_25757 / 2.1474836e9F;
        float res_25759 = 2.0F * res_25758;
        float res_25760 = res_25725 * res_25758;
        float res_25761 = 1.0F + res_25760;
        int64_t res_25762 = fptosi_f32_i64(res_25761);
        float res_25768 = -1.0F + res_25759;
        float res_25769 = sitofp_i64_f32(res_25762);
        float res_25770 = res_25759 * res_25769;
        float res_25743 = fmax32(res_25770, redout_26204);
        
        ((float *) mem_26562)[i_26208] = res_25768;
        ((int64_t *) mem_26564)[i_26208] = res_25762;
        ((float *) mem_26566)[i_26208] = res_25759;
        
        float redout_tmp_26940 = res_25743;
        
        redout_26204 = redout_tmp_26940;
    }
    res_25737 = redout_26204;
    
    float res_25775 = sitofp_i64_f32(steps_25706);
    float dt_25776 = res_25737 / res_25775;
    
    if (mem_26604_cached_sizze_27031 < (size_t) bytes_26561) {
        mem_26604 = realloc(mem_26604, bytes_26561);
        mem_26604_cached_sizze_27031 = bytes_26561;
    }
    if (mem_26606_cached_sizze_27032 < (size_t) bytes_26561) {
        mem_26606 = realloc(mem_26606, bytes_26561);
        mem_26606_cached_sizze_27032 = bytes_26561;
    }
    if (mem_26608_cached_sizze_27033 < (size_t) bytes_26563) {
        mem_26608 = realloc(mem_26608, bytes_26563);
        mem_26608_cached_sizze_27033 = bytes_26563;
    }
    if (mem_26610_cached_sizze_27034 < (size_t) bytes_26561) {
        mem_26610 = realloc(mem_26610, bytes_26561);
        mem_26610_cached_sizze_27034 = bytes_26561;
    }
    for (int64_t i_26222 = 0; i_26222 < numswaps_25707; i_26222++) {
        float res_25786 = ((float *) mem_26566)[i_26222];
        int64_t res_25787 = ((int64_t *) mem_26564)[i_26222];
        bool bounds_invalid_upwards_25789 = slt64(res_25787, 1);
        bool valid_25790 = !bounds_invalid_upwards_25789;
        bool range_valid_c_25791;
        
        if (!valid_25790) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_25787,
                          " is invalid.",
                          "-> #0  cva3d.fut:55:29-48\n   #1  cva3d.fut:96:25-65\n   #2  cva3d.fut:110:16-62\n   #3  cva3d.fut:106:17-110:85\n   #4  cva3d.fut:176:8-67\n   #5  cva3d.fut:170:1-176:75\n");
            return 1;
        }
        
        bool y_25793 = slt64(0, res_25787);
        bool index_certs_25794;
        
        if (!y_25793) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_25787, "].",
                                   "-> #0  cva3d.fut:97:47-70\n   #1  cva3d.fut:110:16-62\n   #2  cva3d.fut:106:17-110:85\n   #3  cva3d.fut:176:8-67\n   #4  cva3d.fut:170:1-176:75\n");
            return 1;
        }
        
        float binop_y_25795 = sitofp_i64_f32(res_25787);
        float index_primexp_25796 = res_25786 * binop_y_25795;
        float negate_arg_25797 = 1.0e-2F * index_primexp_25796;
        float exp_arg_25798 = 0.0F - negate_arg_25797;
        float res_25799 = fpow32(2.7182817F, exp_arg_25798);
        float x_25800 = 1.0F - res_25799;
        float B_25801 = x_25800 / 1.0e-2F;
        float x_25802 = B_25801 - index_primexp_25796;
        float x_25803 = 4.4999997e-6F * x_25802;
        float A1_25804 = x_25803 / 1.0e-4F;
        float y_25805 = fpow32(B_25801, 2.0F);
        float x_25806 = 1.0000001e-6F * y_25805;
        float A2_25807 = x_25806 / 4.0e-2F;
        float exp_arg_25808 = A1_25804 - A2_25807;
        float res_25809 = fpow32(2.7182817F, exp_arg_25808);
        float negate_arg_25810 = 5.0e-2F * B_25801;
        float exp_arg_25811 = 0.0F - negate_arg_25810;
        float res_25812 = fpow32(2.7182817F, exp_arg_25811);
        float res_25813 = res_25809 * res_25812;
        float res_25814;
        float redout_26212 = 0.0F;
        
        for (int64_t i_26213 = 0; i_26213 < res_25787; i_26213++) {
            int64_t index_primexp_26350 = add64(1, i_26213);
            float res_25819 = sitofp_i64_f32(index_primexp_26350);
            float res_25820 = res_25786 * res_25819;
            float negate_arg_25821 = 1.0e-2F * res_25820;
            float exp_arg_25822 = 0.0F - negate_arg_25821;
            float res_25823 = fpow32(2.7182817F, exp_arg_25822);
            float x_25824 = 1.0F - res_25823;
            float B_25825 = x_25824 / 1.0e-2F;
            float x_25826 = B_25825 - res_25820;
            float x_25827 = 4.4999997e-6F * x_25826;
            float A1_25828 = x_25827 / 1.0e-4F;
            float y_25829 = fpow32(B_25825, 2.0F);
            float x_25830 = 1.0000001e-6F * y_25829;
            float A2_25831 = x_25830 / 4.0e-2F;
            float exp_arg_25832 = A1_25828 - A2_25831;
            float res_25833 = fpow32(2.7182817F, exp_arg_25832);
            float negate_arg_25834 = 5.0e-2F * B_25825;
            float exp_arg_25835 = 0.0F - negate_arg_25834;
            float res_25836 = fpow32(2.7182817F, exp_arg_25835);
            float res_25837 = res_25833 * res_25836;
            float res_25817 = res_25837 + redout_26212;
            float redout_tmp_26948 = res_25817;
            
            redout_26212 = redout_tmp_26948;
        }
        res_25814 = redout_26212;
        
        float x_25838 = 1.0F - res_25813;
        float y_25839 = res_25786 * res_25814;
        float res_25840 = x_25838 / y_25839;
        
        ((float *) mem_26604)[i_26222] = res_25840;
        memmove(mem_26606 + i_26222 * 4, mem_26562 + i_26222 * 4,
                (int32_t) sizeof(float));
        memmove(mem_26608 + i_26222 * 8, mem_26564 + i_26222 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_26610 + i_26222 * 4, mem_26566 + i_26222 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_25841 = res_25775 / res_25737;
    bool bounds_invalid_upwards_25842 = slt64(steps_25706, 1);
    bool valid_25843 = !bounds_invalid_upwards_25842;
    bool range_valid_c_25844;
    
    if (!valid_25843) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_25706,
                               " is invalid.",
                               "-> #0  cva3d.fut:61:56-67\n   #1  cva3d.fut:112:17-44\n   #2  cva3d.fut:176:8-67\n   #3  cva3d.fut:170:1-176:75\n");
        return 1;
    }
    
    int64_t bytes_26659 = 4 * steps_25706;
    
    if (mem_26660_cached_sizze_27035 < (size_t) bytes_26659) {
        mem_26660 = realloc(mem_26660, bytes_26659);
        mem_26660_cached_sizze_27035 = bytes_26659;
    }
    for (int64_t i_26229 = 0; i_26229 < steps_25706; i_26229++) {
        int64_t index_primexp_26357 = add64(1, i_26229);
        float res_25848 = sitofp_i64_f32(index_primexp_26357);
        float res_25849 = res_25848 / sims_per_year_25841;
        
        ((float *) mem_26660)[i_26229] = res_25849;
    }
    
    bool bounds_invalid_upwards_25850 = slt64(paths_25705, 0);
    bool valid_25851 = !bounds_invalid_upwards_25850;
    bool range_valid_c_25852;
    
    if (!valid_25851) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_25705,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva3d.fut:116:19-49\n   #4  cva3d.fut:176:8-67\n   #5  cva3d.fut:170:1-176:75\n");
        return 1;
    }
    
    int64_t upper_bound_25855 = sub64(steps_25706, 1);
    float res_25856;
    
    res_25856 = futrts_sqrt32(dt_25776);
    
    int64_t binop_x_26674 = paths_25705 * steps_25706;
    int64_t binop_x_26675 = numswaps_25707 * binop_x_26674;
    int64_t bytes_26673 = 4 * binop_x_26675;
    
    if (mem_26676_cached_sizze_27036 < (size_t) bytes_26673) {
        mem_26676 = realloc(mem_26676, bytes_26673);
        mem_26676_cached_sizze_27036 = bytes_26673;
    }
    if (mem_26680_cached_sizze_27037 < (size_t) bytes_26673) {
        mem_26680 = realloc(mem_26680, bytes_26673);
        mem_26680_cached_sizze_27037 = bytes_26673;
    }
    
    int64_t ctx_val_26704 = steps_25706 * numswaps_25707;
    
    if (mem_26710_cached_sizze_27038 < (size_t) bytes_26659) {
        mem_26710 = realloc(mem_26710, bytes_26659);
        mem_26710_cached_sizze_27038 = bytes_26659;
    }
    if (mem_26724_cached_sizze_27039 < (size_t) bytes_26659) {
        mem_26724 = realloc(mem_26724, bytes_26659);
        mem_26724_cached_sizze_27039 = bytes_26659;
    }
    if (mem_26766_cached_sizze_27040 < (size_t) bytes_26561) {
        mem_26766 = realloc(mem_26766, bytes_26561);
        mem_26766_cached_sizze_27040 = bytes_26561;
    }
    if (mem_26768_cached_sizze_27041 < (size_t) bytes_26561) {
        mem_26768 = realloc(mem_26768, bytes_26561);
        mem_26768_cached_sizze_27041 = bytes_26561;
    }
    for (int64_t i_26246 = 0; i_26246 < paths_25705; i_26246++) {
        int32_t res_25860 = sext_i64_i32(i_26246);
        int32_t x_25861 = lshr32(res_25860, 16);
        int32_t x_25862 = res_25860 ^ x_25861;
        int32_t x_25863 = mul32(73244475, x_25862);
        int32_t x_25864 = lshr32(x_25863, 16);
        int32_t x_25865 = x_25863 ^ x_25864;
        int32_t x_25866 = mul32(73244475, x_25865);
        int32_t x_25867 = lshr32(x_25866, 16);
        int32_t x_25868 = x_25866 ^ x_25867;
        int32_t unsign_arg_25869 = 777822902 ^ x_25868;
        int32_t unsign_arg_25870 = mul32(48271, unsign_arg_25869);
        int32_t unsign_arg_25871 = umod32(unsign_arg_25870, 2147483647);
        
        for (int64_t i_26233 = 0; i_26233 < steps_25706; i_26233++) {
            int32_t res_25874 = sext_i64_i32(i_26233);
            int32_t x_25875 = lshr32(res_25874, 16);
            int32_t x_25876 = res_25874 ^ x_25875;
            int32_t x_25877 = mul32(73244475, x_25876);
            int32_t x_25878 = lshr32(x_25877, 16);
            int32_t x_25879 = x_25877 ^ x_25878;
            int32_t x_25880 = mul32(73244475, x_25879);
            int32_t x_25881 = lshr32(x_25880, 16);
            int32_t x_25882 = x_25880 ^ x_25881;
            int32_t unsign_arg_25883 = unsign_arg_25871 ^ x_25882;
            int32_t unsign_arg_25884 = mul32(48271, unsign_arg_25883);
            int32_t unsign_arg_25885 = umod32(unsign_arg_25884, 2147483647);
            int32_t unsign_arg_25886 = mul32(48271, unsign_arg_25885);
            int32_t unsign_arg_25887 = umod32(unsign_arg_25886, 2147483647);
            float res_25888 = uitofp_i32_f32(unsign_arg_25885);
            float res_25889 = res_25888 / 2.1474836e9F;
            float res_25890 = uitofp_i32_f32(unsign_arg_25887);
            float res_25891 = res_25890 / 2.1474836e9F;
            float res_25892;
            
            res_25892 = futrts_log32(res_25889);
            
            float res_25893 = -2.0F * res_25892;
            float res_25894;
            
            res_25894 = futrts_sqrt32(res_25893);
            
            float res_25895 = 6.2831855F * res_25891;
            float res_25896;
            
            res_25896 = futrts_cos32(res_25895);
            
            float res_25897 = res_25894 * res_25896;
            
            ((float *) mem_26710)[i_26233] = res_25897;
        }
        for (int64_t i_26953 = 0; i_26953 < steps_25706; i_26953++) {
            ((float *) mem_26724)[i_26953] = 5.0e-2F;
        }
        for (int64_t i_25900 = 0; i_25900 < upper_bound_25855; i_25900++) {
            bool y_25902 = slt64(i_25900, steps_25706);
            bool index_certs_25903;
            
            if (!y_25902) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25900,
                              "] out of bounds for array of shape [",
                              steps_25706, "].",
                              "-> #0  cva3d.fut:72:97-104\n   #1  cva3d.fut:124:32-62\n   #2  cva3d.fut:124:22-69\n   #3  cva3d.fut:176:8-67\n   #4  cva3d.fut:170:1-176:75\n");
                return 1;
            }
            
            float shortstep_arg_25904 = ((float *) mem_26710)[i_25900];
            float shortstep_arg_25905 = ((float *) mem_26724)[i_25900];
            float y_25906 = 5.0e-2F - shortstep_arg_25905;
            float x_25907 = 1.0e-2F * y_25906;
            float x_25908 = dt_25776 * x_25907;
            float x_25909 = res_25856 * shortstep_arg_25904;
            float y_25910 = 1.0e-3F * x_25909;
            float delta_r_25911 = x_25908 + y_25910;
            float res_25912 = shortstep_arg_25905 + delta_r_25911;
            int64_t i_25913 = add64(1, i_25900);
            bool x_25914 = sle64(0, i_25913);
            bool y_25915 = slt64(i_25913, steps_25706);
            bool bounds_check_25916 = x_25914 && y_25915;
            bool index_certs_25917;
            
            if (!bounds_check_25916) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25913,
                              "] out of bounds for array of shape [",
                              steps_25706, "].",
                              "-> #0  cva3d.fut:72:58-105\n   #1  cva3d.fut:124:32-62\n   #2  cva3d.fut:124:22-69\n   #3  cva3d.fut:176:8-67\n   #4  cva3d.fut:170:1-176:75\n");
                return 1;
            }
            ((float *) mem_26724)[i_25913] = res_25912;
        }
        for (int64_t i_26239 = 0; i_26239 < steps_25706; i_26239++) {
            float x_25921 = ((float *) mem_26724)[i_26239];
            float x_25922 = ((float *) mem_26660)[i_26239];
            
            for (int64_t i_26957 = 0; i_26957 < numswaps_25707; i_26957++) {
                ((float *) mem_26766)[i_26957] = x_25921;
            }
            for (int64_t i_26958 = 0; i_26958 < numswaps_25707; i_26958++) {
                ((float *) mem_26768)[i_26958] = x_25922;
            }
            memmove(mem_26676 + (i_26246 * ctx_val_26704 + i_26239 *
                                 numswaps_25707) * 4, mem_26766 + 0,
                    numswaps_25707 * (int32_t) sizeof(float));
            memmove(mem_26680 + (i_26246 * ctx_val_26704 + i_26239 *
                                 numswaps_25707) * 4, mem_26768 + 0,
                    numswaps_25707 * (int32_t) sizeof(float));
        }
    }
    
    int64_t flat_dim_25926 = numswaps_25707 * binop_x_26674;
    int64_t bytes_26819 = 8 * flat_dim_25926;
    
    if (mem_26820_cached_sizze_27042 < (size_t) bytes_26819) {
        mem_26820 = realloc(mem_26820, bytes_26819);
        mem_26820_cached_sizze_27042 = bytes_26819;
    }
    if (mem_26822_cached_sizze_27043 < (size_t) bytes_26819) {
        mem_26822 = realloc(mem_26822, bytes_26819);
        mem_26822_cached_sizze_27043 = bytes_26819;
    }
    
    int64_t binop_y_26370 = steps_25706 * numswaps_25707;
    int64_t discard_26257;
    int64_t scanacc_26251 = 0;
    
    for (int64_t i_26254 = 0; i_26254 < flat_dim_25926; i_26254++) {
        int64_t binop_x_26371 = squot64(i_26254, binop_y_26370);
        int64_t binop_y_26373 = binop_y_26370 * binop_x_26371;
        int64_t binop_x_26374 = i_26254 - binop_y_26373;
        int64_t binop_x_26380 = squot64(binop_x_26374, numswaps_25707);
        int64_t binop_y_26381 = numswaps_25707 * binop_x_26380;
        int64_t new_index_26382 = binop_x_26374 - binop_y_26381;
        int64_t x_25937 = ((int64_t *) mem_26608)[new_index_26382];
        float x_25938 = ((float *) mem_26610)[new_index_26382];
        float x_25939 = ((float *) mem_26680)[binop_x_26371 * ctx_val_26704 +
                                              binop_x_26380 * numswaps_25707 +
                                              new_index_26382];
        float x_25940 = x_25939 / x_25938;
        float ceil_arg_25941 = x_25940 - 1.0F;
        float res_25942;
        
        res_25942 = futrts_ceil32(ceil_arg_25941);
        
        int64_t res_25943 = fptosi_f32_i64(res_25942);
        int64_t max_arg_25944 = sub64(x_25937, res_25943);
        int64_t res_25945 = smax64(0, max_arg_25944);
        bool cond_25946 = res_25945 == 0;
        int64_t res_25947;
        
        if (cond_25946) {
            res_25947 = 1;
        } else {
            res_25947 = res_25945;
        }
        
        int64_t res_25936 = add64(res_25947, scanacc_26251);
        
        ((int64_t *) mem_26820)[i_26254] = res_25936;
        ((int64_t *) mem_26822)[i_26254] = res_25947;
        
        int64_t scanacc_tmp_26959 = res_25936;
        
        scanacc_26251 = scanacc_tmp_26959;
    }
    discard_26257 = scanacc_26251;
    
    int64_t res_25950;
    int64_t redout_26258 = 0;
    
    for (int64_t i_26259 = 0; i_26259 < flat_dim_25926; i_26259++) {
        int64_t x_25954 = ((int64_t *) mem_26822)[i_26259];
        int64_t res_25953 = add64(x_25954, redout_26258);
        int64_t redout_tmp_26962 = res_25953;
        
        redout_26258 = redout_tmp_26962;
    }
    res_25950 = redout_26258;
    
    bool bounds_invalid_upwards_25955 = slt64(res_25950, 0);
    bool valid_25956 = !bounds_invalid_upwards_25955;
    bool range_valid_c_25957;
    
    if (!valid_25956) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", res_25950,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:176:8-67\n   #7  cva3d.fut:170:1-176:75\n");
        return 1;
    }
    
    int64_t bytes_26847 = 8 * res_25950;
    
    if (mem_26848_cached_sizze_27044 < (size_t) bytes_26847) {
        mem_26848 = realloc(mem_26848, bytes_26847);
        mem_26848_cached_sizze_27044 = bytes_26847;
    }
    for (int64_t i_26963 = 0; i_26963 < res_25950; i_26963++) {
        ((int64_t *) mem_26848)[i_26963] = 0;
    }
    for (int64_t iter_26260 = 0; iter_26260 < flat_dim_25926; iter_26260++) {
        int64_t i_p_o_26427 = add64(-1, iter_26260);
        int64_t rot_i_26428 = smod64(i_p_o_26427, flat_dim_25926);
        int64_t pixel_26263 = ((int64_t *) mem_26820)[rot_i_26428];
        bool cond_25965 = iter_26260 == 0;
        int64_t res_25966;
        
        if (cond_25965) {
            res_25966 = 0;
        } else {
            res_25966 = pixel_26263;
        }
        
        bool less_than_zzero_26264 = slt64(res_25966, 0);
        bool greater_than_sizze_26265 = sle64(res_25950, res_25966);
        bool outside_bounds_dim_26266 = less_than_zzero_26264 ||
             greater_than_sizze_26265;
        
        if (!outside_bounds_dim_26266) {
            int64_t read_hist_26268 = ((int64_t *) mem_26848)[res_25966];
            int64_t res_25962 = smax64(iter_26260, read_hist_26268);
            
            ((int64_t *) mem_26848)[res_25966] = res_25962;
        }
    }
    if (mem_26862_cached_sizze_27045 < (size_t) bytes_26847) {
        mem_26862 = realloc(mem_26862, bytes_26847);
        mem_26862_cached_sizze_27045 = bytes_26847;
    }
    
    int64_t discard_26281;
    int64_t scanacc_26274 = 0;
    
    for (int64_t i_26277 = 0; i_26277 < res_25950; i_26277++) {
        int64_t x_25976 = ((int64_t *) mem_26848)[i_26277];
        bool res_25977 = slt64(0, x_25976);
        int64_t res_25974;
        
        if (res_25977) {
            res_25974 = x_25976;
        } else {
            int64_t res_25975 = add64(x_25976, scanacc_26274);
            
            res_25974 = res_25975;
        }
        ((int64_t *) mem_26862)[i_26277] = res_25974;
        
        int64_t scanacc_tmp_26965 = res_25974;
        
        scanacc_26274 = scanacc_tmp_26965;
    }
    discard_26281 = scanacc_26274;
    
    int64_t bytes_26875 = 4 * res_25950;
    
    if (mem_26876_cached_sizze_27046 < (size_t) bytes_26875) {
        mem_26876 = realloc(mem_26876, bytes_26875);
        mem_26876_cached_sizze_27046 = bytes_26875;
    }
    if (mem_26878_cached_sizze_27047 < (size_t) res_25950) {
        mem_26878 = realloc(mem_26878, res_25950);
        mem_26878_cached_sizze_27047 = res_25950;
    }
    
    int64_t lstel_tmp_26029 = 1;
    int64_t inpacc_25987;
    float inpacc_25989;
    int64_t inpacc_25995;
    float inpacc_25997;
    
    inpacc_25995 = 0;
    inpacc_25997 = 0.0F;
    for (int64_t i_26320 = 0; i_26320 < res_25950; i_26320++) {
        int64_t x_26442 = ((int64_t *) mem_26862)[i_26320];
        int64_t i_p_o_26444 = add64(-1, i_26320);
        int64_t rot_i_26445 = smod64(i_p_o_26444, res_25950);
        int64_t x_26446 = ((int64_t *) mem_26862)[rot_i_26445];
        bool res_26447 = x_26442 == x_26446;
        bool res_26448 = !res_26447;
        int64_t res_26031;
        
        if (res_26448) {
            res_26031 = lstel_tmp_26029;
        } else {
            int64_t res_26032 = add64(1, inpacc_25995);
            
            res_26031 = res_26032;
        }
        
        int64_t res_26046;
        
        if (res_26448) {
            res_26046 = ctx_val_26583;
        } else {
            int64_t res_26047 = add64(1, inpacc_25995);
            
            res_26046 = res_26047;
        }
        
        int64_t res_26048 = sub64(res_26046, 1);
        bool x_26463 = sle64(0, x_26442);
        bool y_26464 = slt64(x_26442, flat_dim_25926);
        bool bounds_check_26465 = x_26463 && y_26464;
        bool index_certs_26466;
        
        if (!bounds_check_26465) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", x_26442,
                                   "] out of bounds for array of shape [",
                                   flat_dim_25926, "].",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:176:8-67\n   #7  cva3d.fut:170:1-176:75\n");
            return 1;
        }
        
        int64_t new_index_26467 = squot64(x_26442, binop_y_26370);
        int64_t binop_y_26468 = binop_y_26370 * new_index_26467;
        int64_t binop_x_26469 = x_26442 - binop_y_26468;
        int64_t new_index_26470 = squot64(binop_x_26469, numswaps_25707);
        int64_t binop_y_26471 = numswaps_25707 * new_index_26470;
        int64_t new_index_26472 = binop_x_26469 - binop_y_26471;
        float lifted_0_get_arg_26473 = ((float *) mem_26676)[new_index_26467 *
                                                             ctx_val_26704 +
                                                             new_index_26470 *
                                                             numswaps_25707 +
                                                             new_index_26472];
        float lifted_0_get_arg_26474 = ((float *) mem_26604)[new_index_26472];
        float lifted_0_get_arg_26475 = ((float *) mem_26606)[new_index_26472];
        int64_t lifted_0_get_arg_26476 =
                ((int64_t *) mem_26608)[new_index_26472];
        float lifted_0_get_arg_26477 = ((float *) mem_26610)[new_index_26472];
        float lifted_0_get_arg_26478 = ((float *) mem_26680)[new_index_26467 *
                                                             ctx_val_26704 +
                                                             new_index_26470 *
                                                             numswaps_25707 +
                                                             new_index_26472];
        float x_26479 = lifted_0_get_arg_26478 / lifted_0_get_arg_26477;
        float ceil_arg_26480 = x_26479 - 1.0F;
        float res_26481;
        
        res_26481 = futrts_ceil32(ceil_arg_26480);
        
        int64_t res_26482 = fptosi_f32_i64(res_26481);
        int64_t max_arg_26483 = sub64(lifted_0_get_arg_26476, res_26482);
        int64_t res_26484 = smax64(0, max_arg_26483);
        bool cond_26485 = res_26484 == 0;
        float res_26486;
        
        if (cond_26485) {
            res_26486 = 0.0F;
        } else {
            float res_26487;
            
            res_26487 = futrts_ceil32(x_26479);
            
            float start_26488 = lifted_0_get_arg_26477 * res_26487;
            float res_26489;
            
            res_26489 = futrts_ceil32(ceil_arg_26480);
            
            int64_t res_26490 = fptosi_f32_i64(res_26489);
            int64_t max_arg_26491 = sub64(lifted_0_get_arg_26476, res_26490);
            int64_t res_26492 = smax64(0, max_arg_26491);
            int64_t sizze_26493 = sub64(res_26492, 1);
            bool cond_26494 = res_26048 == 0;
            float res_26495;
            
            if (cond_26494) {
                res_26495 = 1.0F;
            } else {
                res_26495 = 0.0F;
            }
            
            bool cond_26496 = slt64(0, res_26048);
            float res_26497;
            
            if (cond_26496) {
                float y_26498 = lifted_0_get_arg_26474 * lifted_0_get_arg_26477;
                float res_26499 = res_26495 - y_26498;
                
                res_26497 = res_26499;
            } else {
                res_26497 = res_26495;
            }
            
            bool cond_26500 = res_26048 == sizze_26493;
            float res_26501;
            
            if (cond_26500) {
                float res_26502 = res_26497 - 1.0F;
                
                res_26501 = res_26502;
            } else {
                res_26501 = res_26497;
            }
            
            float res_26503 = lifted_0_get_arg_26475 * res_26501;
            float res_26504 = sitofp_i64_f32(res_26048);
            float y_26505 = lifted_0_get_arg_26477 * res_26504;
            float bondprice_arg_26506 = start_26488 + y_26505;
            float y_26507 = bondprice_arg_26506 - lifted_0_get_arg_26478;
            float negate_arg_26508 = 1.0e-2F * y_26507;
            float exp_arg_26509 = 0.0F - negate_arg_26508;
            float res_26510 = fpow32(2.7182817F, exp_arg_26509);
            float x_26511 = 1.0F - res_26510;
            float B_26512 = x_26511 / 1.0e-2F;
            float x_26513 = B_26512 - bondprice_arg_26506;
            float x_26514 = lifted_0_get_arg_26478 + x_26513;
            float x_26515 = 4.4999997e-6F * x_26514;
            float A1_26516 = x_26515 / 1.0e-4F;
            float y_26517 = fpow32(B_26512, 2.0F);
            float x_26518 = 1.0000001e-6F * y_26517;
            float A2_26519 = x_26518 / 4.0e-2F;
            float exp_arg_26520 = A1_26516 - A2_26519;
            float res_26521 = fpow32(2.7182817F, exp_arg_26520);
            float negate_arg_26522 = lifted_0_get_arg_26473 * B_26512;
            float exp_arg_26523 = 0.0F - negate_arg_26522;
            float res_26524 = fpow32(2.7182817F, exp_arg_26523);
            float res_26525 = res_26521 * res_26524;
            float res_26526 = res_26503 * res_26525;
            
            res_26486 = res_26526;
        }
        
        float lstel_tmp_26116 = res_26486;
        float res_26122;
        
        if (res_26448) {
            res_26122 = res_26486;
        } else {
            float res_26123 = inpacc_25997 + res_26486;
            
            res_26122 = res_26123;
        }
        
        float res_26125;
        
        if (res_26448) {
            res_26125 = lstel_tmp_26116;
        } else {
            float res_26126 = inpacc_25997 + res_26486;
            
            res_26125 = res_26126;
        }
        ((float *) mem_26876)[i_26320] = res_26122;
        ((bool *) mem_26878)[i_26320] = res_26448;
        
        int64_t inpacc_tmp_26967 = res_26031;
        float inpacc_tmp_26968 = res_26125;
        
        inpacc_25995 = inpacc_tmp_26967;
        inpacc_25997 = inpacc_tmp_26968;
    }
    inpacc_25987 = inpacc_25995;
    inpacc_25989 = inpacc_25997;
    if (mem_26904_cached_sizze_27048 < (size_t) bytes_26847) {
        mem_26904 = realloc(mem_26904, bytes_26847);
        mem_26904_cached_sizze_27048 = bytes_26847;
    }
    
    int64_t discard_26329;
    int64_t scanacc_26325 = 0;
    
    for (int64_t i_26327 = 0; i_26327 < res_25950; i_26327++) {
        int64_t i_p_o_26537 = add64(1, i_26327);
        int64_t rot_i_26538 = smod64(i_p_o_26537, res_25950);
        bool x_26139 = ((bool *) mem_26878)[rot_i_26538];
        int64_t res_26140 = btoi_bool_i64(x_26139);
        int64_t res_26138 = add64(res_26140, scanacc_26325);
        
        ((int64_t *) mem_26904)[i_26327] = res_26138;
        
        int64_t scanacc_tmp_26971 = res_26138;
        
        scanacc_26325 = scanacc_tmp_26971;
    }
    discard_26329 = scanacc_26325;
    
    bool cond_26141 = slt64(0, res_25950);
    int64_t num_segments_26142;
    
    if (cond_26141) {
        int64_t i_26143 = sub64(res_25950, 1);
        bool x_26144 = sle64(0, i_26143);
        bool y_26145 = slt64(i_26143, res_25950);
        bool bounds_check_26146 = x_26144 && y_26145;
        bool index_certs_26147;
        
        if (!bounds_check_26146) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", i_26143,
                                   "] out of bounds for array of shape [",
                                   res_25950, "].",
                                   "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva3d.fut:135:18-77\n   #5  cva3d.fut:176:8-67\n   #6  cva3d.fut:170:1-176:75\n");
            return 1;
        }
        
        int64_t res_26148 = ((int64_t *) mem_26904)[i_26143];
        
        num_segments_26142 = res_26148;
    } else {
        num_segments_26142 = 0;
    }
    
    bool bounds_invalid_upwards_26149 = slt64(num_segments_26142, 0);
    bool valid_26150 = !bounds_invalid_upwards_26149;
    bool range_valid_c_26151;
    
    if (!valid_26150) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", num_segments_26142,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva3d.fut:135:18-77\n   #6  cva3d.fut:176:8-67\n   #7  cva3d.fut:170:1-176:75\n");
        return 1;
    }
    
    int64_t bytes_26917 = 4 * num_segments_26142;
    
    if (mem_26918_cached_sizze_27049 < (size_t) bytes_26917) {
        mem_26918 = realloc(mem_26918, bytes_26917);
        mem_26918_cached_sizze_27049 = bytes_26917;
    }
    for (int64_t i_26973 = 0; i_26973 < num_segments_26142; i_26973++) {
        ((float *) mem_26918)[i_26973] = 0.0F;
    }
    for (int64_t write_iter_26330 = 0; write_iter_26330 < res_25950;
         write_iter_26330++) {
        int64_t write_iv_26332 = ((int64_t *) mem_26904)[write_iter_26330];
        int64_t i_p_o_26540 = add64(1, write_iter_26330);
        int64_t rot_i_26541 = smod64(i_p_o_26540, res_25950);
        bool write_iv_26333 = ((bool *) mem_26878)[rot_i_26541];
        int64_t res_26157;
        
        if (write_iv_26333) {
            int64_t res_26158 = sub64(write_iv_26332, 1);
            
            res_26157 = res_26158;
        } else {
            res_26157 = -1;
        }
        
        bool less_than_zzero_26335 = slt64(res_26157, 0);
        bool greater_than_sizze_26336 = sle64(num_segments_26142, res_26157);
        bool outside_bounds_dim_26337 = less_than_zzero_26335 ||
             greater_than_sizze_26336;
        
        if (!outside_bounds_dim_26337) {
            memmove(mem_26918 + res_26157 * 4, mem_26876 + write_iter_26330 * 4,
                    (int32_t) sizeof(float));
        }
    }
    
    bool dim_match_26159 = flat_dim_25926 == num_segments_26142;
    bool empty_or_match_cert_26160;
    
    if (!dim_match_26159) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Value of (core language) shape (",
                               num_segments_26142,
                               ") cannot match shape of type `[",
                               flat_dim_25926, "]b`.",
                               "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva3d.fut:135:18-77\n   #2  cva3d.fut:176:8-67\n   #3  cva3d.fut:170:1-176:75\n");
        return 1;
    }
    
    float res_26161 = sitofp_i64_f32(paths_25705);
    float res_26164;
    float redout_26345 = 0.0F;
    
    for (int64_t i_26346 = 0; i_26346 < steps_25706; i_26346++) {
        float x_26169 = ((float *) mem_26660)[i_26346];
        int64_t binop_y_26544 = numswaps_25707 * i_26346;
        float res_26170;
        float redout_26343 = 0.0F;
        
        for (int64_t i_26344 = 0; i_26344 < paths_25705; i_26344++) {
            int64_t binop_x_26543 = i_26344 * binop_y_26370;
            int64_t binop_x_26545 = binop_x_26543 + binop_y_26544;
            float res_26175;
            float redout_26341 = 0.0F;
            
            for (int64_t i_26342 = 0; i_26342 < numswaps_25707; i_26342++) {
                int64_t new_index_26546 = i_26342 + binop_x_26545;
                float x_26179 = ((float *) mem_26918)[new_index_26546];
                float res_26178 = x_26179 + redout_26341;
                float redout_tmp_26977 = res_26178;
                
                redout_26341 = redout_tmp_26977;
            }
            res_26175 = redout_26341;
            
            float res_26180 = fmax32(0.0F, res_26175);
            float res_26173 = res_26180 + redout_26343;
            float redout_tmp_26976 = res_26173;
            
            redout_26343 = redout_tmp_26976;
        }
        res_26170 = redout_26343;
        
        float res_26181 = res_26170 / res_26161;
        float negate_arg_26182 = 1.0e-2F * x_26169;
        float exp_arg_26183 = 0.0F - negate_arg_26182;
        float res_26184 = fpow32(2.7182817F, exp_arg_26183);
        float x_26185 = 1.0F - res_26184;
        float B_26186 = x_26185 / 1.0e-2F;
        float x_26187 = B_26186 - x_26169;
        float x_26188 = 4.4999997e-6F * x_26187;
        float A1_26189 = x_26188 / 1.0e-4F;
        float y_26190 = fpow32(B_26186, 2.0F);
        float x_26191 = 1.0000001e-6F * y_26190;
        float A2_26192 = x_26191 / 4.0e-2F;
        float exp_arg_26193 = A1_26189 - A2_26192;
        float res_26194 = fpow32(2.7182817F, exp_arg_26193);
        float negate_arg_26195 = 5.0e-2F * B_26186;
        float exp_arg_26196 = 0.0F - negate_arg_26195;
        float res_26197 = fpow32(2.7182817F, exp_arg_26196);
        float res_26198 = res_26194 * res_26197;
        float res_26199 = res_26181 * res_26198;
        float res_26167 = res_26199 + redout_26345;
        float redout_tmp_26975 = res_26167;
        
        redout_26345 = redout_tmp_26975;
    }
    res_26164 = redout_26345;
    
    float CVA_26200 = 6.0e-3F * res_26164;
    
    scalar_out_26939 = CVA_26200;
    *out_scalar_out_27027 = scalar_out_26939;
    
  cleanup:
    { }
    free(mem_26562);
    free(mem_26564);
    free(mem_26566);
    free(mem_26604);
    free(mem_26606);
    free(mem_26608);
    free(mem_26610);
    free(mem_26660);
    free(mem_26676);
    free(mem_26680);
    free(mem_26710);
    free(mem_26724);
    free(mem_26766);
    free(mem_26768);
    free(mem_26820);
    free(mem_26822);
    free(mem_26848);
    free(mem_26862);
    free(mem_26876);
    free(mem_26878);
    free(mem_26904);
    free(mem_26918);
    return err;
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
                       struct futhark_f32_1d **out1, const int64_t in0, const
                       int64_t in1, const struct futhark_f32_1d *in2, const
                       struct futhark_i64_1d *in3, const
                       struct futhark_f32_1d *in4, const float in5, const
                       float in6, const float in7, const float in8)
{
    struct memblock swap_term_mem_26561;
    
    swap_term_mem_26561.references = NULL;
    
    struct memblock payments_mem_26562;
    
    payments_mem_26562.references = NULL;
    
    struct memblock notional_mem_26563;
    
    notional_mem_26563.references = NULL;
    
    int64_t n_24782;
    int64_t n_24783;
    int64_t n_24784;
    int64_t paths_24785;
    int64_t steps_24786;
    float a_24790;
    float b_24791;
    float sigma_24792;
    float r0_24793;
    float scalar_out_26939;
    struct memblock out_mem_26940;
    
    out_mem_26940.references = NULL;
    
    int64_t out_arrsizze_26941;
    
    lock_lock(&ctx->lock);
    paths_24785 = in0;
    steps_24786 = in1;
    swap_term_mem_26561 = in2->mem;
    n_24782 = in2->shape[0];
    payments_mem_26562 = in3->mem;
    n_24783 = in3->shape[0];
    notional_mem_26563 = in4->mem;
    n_24784 = in4->shape[0];
    a_24790 = in5;
    b_24791 = in6;
    sigma_24792 = in7;
    r0_24793 = in8;
    
    int ret = futrts_main(ctx, &scalar_out_26939, &out_mem_26940,
                          &out_arrsizze_26941, swap_term_mem_26561,
                          payments_mem_26562, notional_mem_26563, n_24782,
                          n_24783, n_24784, paths_24785, steps_24786, a_24790,
                          b_24791, sigma_24792, r0_24793);
    
    if (ret == 0) {
        *out0 = scalar_out_26939;
        assert((*out1 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out1)->mem = out_mem_26940;
        (*out1)->shape[0] = out_arrsizze_26941;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test(struct futhark_context *ctx, float *out0, const
                       int64_t in0, const int64_t in1)
{
    int64_t paths_25265;
    int64_t steps_25266;
    float scalar_out_26939;
    
    lock_lock(&ctx->lock);
    paths_25265 = in0;
    steps_25266 = in1;
    
    int ret = futrts_test(ctx, &scalar_out_26939, paths_25265, steps_25266);
    
    if (ret == 0) {
        *out0 = scalar_out_26939;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test2(struct futhark_context *ctx, float *out0, const
                        int64_t in0, const int64_t in1, const int64_t in2)
{
    int64_t paths_25705;
    int64_t steps_25706;
    int64_t numswaps_25707;
    float scalar_out_26939;
    
    lock_lock(&ctx->lock);
    paths_25705 = in0;
    steps_25706 = in1;
    numswaps_25707 = in2;
    
    int ret = futrts_test2(ctx, &scalar_out_26939, paths_25705, steps_25706,
                           numswaps_25707);
    
    if (ret == 0) {
        *out0 = scalar_out_26939;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
