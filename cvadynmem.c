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
    
    int64_t read_value_28820;
    
    if (read_scalar(&i64_info, &read_value_28820) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28821;
    
    if (read_scalar(&i64_info, &read_value_28821) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_28822;
    int64_t read_shape_28823[1];
    float *read_arr_28824 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_28824, read_shape_28823, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
                      "[]", f32_info.type_name, strerror(errno));
    
    struct futhark_i64_1d *read_value_28825;
    int64_t read_shape_28826[1];
    int64_t *read_arr_28827 = NULL;
    
    errno = 0;
    if (read_array(&i64_info, (void **) &read_arr_28827, read_shape_28826, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
                      "[]", i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_28828;
    int64_t read_shape_28829[1];
    float *read_arr_28830 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_28830, read_shape_28829, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4,
                      "[]", f32_info.type_name, strerror(errno));
    
    float read_value_28831;
    
    if (read_scalar(&f32_info, &read_value_28831) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      5, f32_info.type_name, strerror(errno));
    
    float read_value_28832;
    
    if (read_scalar(&f32_info, &read_value_28832) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      6, f32_info.type_name, strerror(errno));
    
    float read_value_28833;
    
    if (read_scalar(&f32_info, &read_value_28833) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      7, f32_info.type_name, strerror(errno));
    
    float read_value_28834;
    
    if (read_scalar(&f32_info, &read_value_28834) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      8, f32_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"main\"");
    
    float result_28835;
    struct futhark_f32_1d *result_28836;
    
    if (perform_warmup) {
        int r;
        
        ;
        ;
        assert((read_value_28822 = futhark_new_f32_1d(ctx, read_arr_28824,
                                                      read_shape_28823[0])) !=
            0);
        assert((read_value_28825 = futhark_new_i64_1d(ctx, read_arr_28827,
                                                      read_shape_28826[0])) !=
            0);
        assert((read_value_28828 = futhark_new_f32_1d(ctx, read_arr_28830,
                                                      read_shape_28829[0])) !=
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
        r = futhark_entry_main(ctx, &result_28835, &result_28836,
                               read_value_28820, read_value_28821,
                               read_value_28822, read_value_28825,
                               read_value_28828, read_value_28831,
                               read_value_28832, read_value_28833,
                               read_value_28834);
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
        assert(futhark_free_f32_1d(ctx, read_value_28822) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_28825) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_28828) == 0);
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_f32_1d(ctx, result_28836) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        ;
        assert((read_value_28822 = futhark_new_f32_1d(ctx, read_arr_28824,
                                                      read_shape_28823[0])) !=
            0);
        assert((read_value_28825 = futhark_new_i64_1d(ctx, read_arr_28827,
                                                      read_shape_28826[0])) !=
            0);
        assert((read_value_28828 = futhark_new_f32_1d(ctx, read_arr_28830,
                                                      read_shape_28829[0])) !=
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
        r = futhark_entry_main(ctx, &result_28835, &result_28836,
                               read_value_28820, read_value_28821,
                               read_value_28822, read_value_28825,
                               read_value_28828, read_value_28831,
                               read_value_28832, read_value_28833,
                               read_value_28834);
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
        assert(futhark_free_f32_1d(ctx, read_value_28822) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_28825) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_28828) == 0);
        ;
        ;
        ;
        ;
        if (run < num_runs - 1) {
            ;
            assert(futhark_free_f32_1d(ctx, result_28836) == 0);
        }
    }
    ;
    ;
    free(read_arr_28824);
    free(read_arr_28827);
    free(read_arr_28830);
    ;
    ;
    ;
    ;
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &f32_info, &result_28835);
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_28836)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_28836, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_28836), 1);
        free(arr);
    }
    printf("\n");
    ;
    assert(futhark_free_f32_1d(ctx, result_28836) == 0);
}
static void futrts_cli_entry_test(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    int64_t read_value_28837;
    
    if (read_scalar(&i64_info, &read_value_28837) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28838;
    
    if (read_scalar(&i64_info, &read_value_28838) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"test\"");
    
    float result_28839;
    
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
        r = futhark_entry_test(ctx, &result_28839, read_value_28837,
                               read_value_28838);
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
        r = futhark_entry_test(ctx, &result_28839, read_value_28837,
                               read_value_28838);
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
    write_scalar(stdout, binary_output, &f32_info, &result_28839);
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
    
    int64_t read_value_28840;
    
    if (read_scalar(&i64_info, &read_value_28840) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28841;
    
    if (read_scalar(&i64_info, &read_value_28841) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    
    int64_t read_value_28842;
    
    if (read_scalar(&i64_info, &read_value_28842) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      2, i64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"test2\"");
    
    float result_28843;
    
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
        r = futhark_entry_test2(ctx, &result_28843, read_value_28840,
                                read_value_28841, read_value_28842);
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
        r = futhark_entry_test2(ctx, &result_28843, read_value_28840,
                                read_value_28841, read_value_28842);
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
    write_scalar(stdout, binary_output, &f32_info, &result_28843);
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
static float testzistatic_array_realtype_28805[45] = {1.0F, -0.5F, 1.0F, 1.0F,
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
static int64_t testzistatic_array_realtype_28806[45] = {10, 20, 5, 5, 50, 20,
                                                        30, 15, 18, 10, 200, 5,
                                                        5, 50, 20, 30, 15, 18,
                                                        10, 20, 5, 5, 100, 20,
                                                        30, 15, 18, 10, 20, 5,
                                                        5, 50, 20, 30, 15, 18,
                                                        10, 20, 5, 5, 50, 20,
                                                        30, 15, 18};
static float testzistatic_array_realtype_28807[45] = {1.0F, 0.5F, 0.25F, 0.1F,
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
    struct memblock testzistatic_array_28763;
    struct memblock testzistatic_array_28764;
    struct memblock testzistatic_array_28765;
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
    ctx->testzistatic_array_28763 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_28805,
                                                       0};
    ctx->testzistatic_array_28764 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_28806,
                                                       0};
    ctx->testzistatic_array_28765 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_28807,
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
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_28781,
                       struct memblock *out_mem_p_28782,
                       int64_t *out_out_arrsizze_28783,
                       struct memblock swap_term_mem_28592,
                       struct memblock payments_mem_28593,
                       struct memblock notional_mem_28594, int64_t n_27423,
                       int64_t n_27424, int64_t n_27425, int64_t paths_27426,
                       int64_t steps_27427, float a_27431, float b_27432,
                       float sigma_27433, float r0_27434);
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_28793,
                       int64_t paths_27863, int64_t steps_27864);
static int futrts_test2(struct futhark_context *ctx,
                        float *out_scalar_out_28808, int64_t paths_28162,
                        int64_t steps_28163, int64_t numswaps_28164);
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
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_28781,
                       struct memblock *out_mem_p_28782,
                       int64_t *out_out_arrsizze_28783,
                       struct memblock swap_term_mem_28592,
                       struct memblock payments_mem_28593,
                       struct memblock notional_mem_28594, int64_t n_27423,
                       int64_t n_27424, int64_t n_27425, int64_t paths_27426,
                       int64_t steps_27427, float a_27431, float b_27432,
                       float sigma_27433, float r0_27434)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_28596_cached_sizze_28784 = 0;
    char *mem_28596 = NULL;
    size_t mem_28598_cached_sizze_28785 = 0;
    char *mem_28598 = NULL;
    size_t mem_28600_cached_sizze_28786 = 0;
    char *mem_28600 = NULL;
    size_t mem_28602_cached_sizze_28787 = 0;
    char *mem_28602 = NULL;
    size_t mem_28652_cached_sizze_28788 = 0;
    char *mem_28652 = NULL;
    size_t mem_28667_cached_sizze_28789 = 0;
    char *mem_28667 = NULL;
    size_t mem_28679_cached_sizze_28790 = 0;
    char *mem_28679 = NULL;
    size_t mem_28693_cached_sizze_28791 = 0;
    char *mem_28693 = NULL;
    size_t mem_28721_cached_sizze_28792 = 0;
    char *mem_28721 = NULL;
    float scalar_out_28762;
    struct memblock out_mem_28763;
    
    out_mem_28763.references = NULL;
    
    int64_t out_arrsizze_28764;
    bool dim_match_27435 = n_27423 == n_27424;
    bool empty_or_match_cert_27436;
    
    if (!dim_match_27435) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cvadynmem.fut:111:1-166:19\n");
        if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_27437 = n_27423 == n_27425;
    bool empty_or_match_cert_27438;
    
    if (!dim_match_27437) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cvadynmem.fut:111:1-166:19\n");
        if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
            return 1;
        return 1;
    }
    
    float res_27440;
    float redout_28517 = -INFINITY;
    
    for (int64_t i_28518 = 0; i_28518 < n_27423; i_28518++) {
        float x_27444 = ((float *) swap_term_mem_28592.mem)[i_28518];
        int64_t x_27445 = ((int64_t *) payments_mem_28593.mem)[i_28518];
        float res_27446 = sitofp_i64_f32(x_27445);
        float res_27447 = x_27444 * res_27446;
        float res_27443 = fmax32(res_27447, redout_28517);
        float redout_tmp_28765 = res_27443;
        
        redout_28517 = redout_tmp_28765;
    }
    res_27440 = redout_28517;
    
    float res_27448 = sitofp_i64_f32(steps_27427);
    float dt_27449 = res_27440 / res_27448;
    float x_27451 = fpow32(a_27431, 2.0F);
    float x_27452 = b_27432 * x_27451;
    float x_27453 = fpow32(sigma_27433, 2.0F);
    float y_27454 = x_27453 / 2.0F;
    float y_27455 = x_27452 - y_27454;
    float y_27456 = 4.0F * a_27431;
    int64_t bytes_28595 = 4 * n_27423;
    
    if (mem_28596_cached_sizze_28784 < (size_t) bytes_28595) {
        mem_28596 = realloc(mem_28596, bytes_28595);
        mem_28596_cached_sizze_28784 = bytes_28595;
    }
    if (mem_28598_cached_sizze_28785 < (size_t) bytes_28595) {
        mem_28598 = realloc(mem_28598, bytes_28595);
        mem_28598_cached_sizze_28785 = bytes_28595;
    }
    
    int64_t bytes_28599 = 8 * n_27423;
    
    if (mem_28600_cached_sizze_28786 < (size_t) bytes_28599) {
        mem_28600 = realloc(mem_28600, bytes_28599);
        mem_28600_cached_sizze_28786 = bytes_28599;
    }
    if (mem_28602_cached_sizze_28787 < (size_t) bytes_28595) {
        mem_28602 = realloc(mem_28602, bytes_28595);
        mem_28602_cached_sizze_28787 = bytes_28595;
    }
    for (int64_t i_28529 = 0; i_28529 < n_27423; i_28529++) {
        float res_27466 = ((float *) swap_term_mem_28592.mem)[i_28529];
        int64_t res_27467 = ((int64_t *) payments_mem_28593.mem)[i_28529];
        bool bounds_invalid_upwards_27469 = slt64(res_27467, 1);
        bool valid_27470 = !bounds_invalid_upwards_27469;
        bool range_valid_c_27471;
        
        if (!valid_27470) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_27467,
                          " is invalid.",
                          "-> #0  cvadynmem.fut:64:29-48\n   #1  cvadynmem.fut:105:25-65\n   #2  cvadynmem.fut:120:16-62\n   #3  cvadynmem.fut:116:17-120:85\n   #4  cvadynmem.fut:111:1-166:19\n");
            if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
                return 1;
            return 1;
        }
        
        bool y_27473 = slt64(0, res_27467);
        bool index_certs_27474;
        
        if (!y_27473) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_27467, "].",
                                   "-> #0  cvadynmem.fut:106:47-70\n   #1  cvadynmem.fut:120:16-62\n   #2  cvadynmem.fut:116:17-120:85\n   #3  cvadynmem.fut:111:1-166:19\n");
            if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
                return 1;
            return 1;
        }
        
        float binop_y_27475 = sitofp_i64_f32(res_27467);
        float index_primexp_27476 = res_27466 * binop_y_27475;
        float negate_arg_27477 = a_27431 * index_primexp_27476;
        float exp_arg_27478 = 0.0F - negate_arg_27477;
        float res_27479 = fpow32(2.7182817F, exp_arg_27478);
        float x_27480 = 1.0F - res_27479;
        float B_27481 = x_27480 / a_27431;
        float x_27482 = B_27481 - index_primexp_27476;
        float x_27483 = y_27455 * x_27482;
        float A1_27484 = x_27483 / x_27451;
        float y_27485 = fpow32(B_27481, 2.0F);
        float x_27486 = x_27453 * y_27485;
        float A2_27487 = x_27486 / y_27456;
        float exp_arg_27488 = A1_27484 - A2_27487;
        float res_27489 = fpow32(2.7182817F, exp_arg_27488);
        float negate_arg_27490 = r0_27434 * B_27481;
        float exp_arg_27491 = 0.0F - negate_arg_27490;
        float res_27492 = fpow32(2.7182817F, exp_arg_27491);
        float res_27493 = res_27489 * res_27492;
        float res_27494;
        float redout_28519 = 0.0F;
        
        for (int64_t i_28520 = 0; i_28520 < res_27467; i_28520++) {
            int64_t index_primexp_28564 = add64(1, i_28520);
            float res_27499 = sitofp_i64_f32(index_primexp_28564);
            float res_27500 = res_27466 * res_27499;
            float negate_arg_27501 = a_27431 * res_27500;
            float exp_arg_27502 = 0.0F - negate_arg_27501;
            float res_27503 = fpow32(2.7182817F, exp_arg_27502);
            float x_27504 = 1.0F - res_27503;
            float B_27505 = x_27504 / a_27431;
            float x_27506 = B_27505 - res_27500;
            float x_27507 = y_27455 * x_27506;
            float A1_27508 = x_27507 / x_27451;
            float y_27509 = fpow32(B_27505, 2.0F);
            float x_27510 = x_27453 * y_27509;
            float A2_27511 = x_27510 / y_27456;
            float exp_arg_27512 = A1_27508 - A2_27511;
            float res_27513 = fpow32(2.7182817F, exp_arg_27512);
            float negate_arg_27514 = r0_27434 * B_27505;
            float exp_arg_27515 = 0.0F - negate_arg_27514;
            float res_27516 = fpow32(2.7182817F, exp_arg_27515);
            float res_27517 = res_27513 * res_27516;
            float res_27497 = res_27517 + redout_28519;
            float redout_tmp_28770 = res_27497;
            
            redout_28519 = redout_tmp_28770;
        }
        res_27494 = redout_28519;
        
        float x_27518 = 1.0F - res_27493;
        float y_27519 = res_27466 * res_27494;
        float res_27520 = x_27518 / y_27519;
        
        ((float *) mem_28596)[i_28529] = res_27520;
        memmove(mem_28598 + i_28529 * 4, notional_mem_28594.mem + i_28529 * 4,
                (int32_t) sizeof(float));
        memmove(mem_28600 + i_28529 * 8, payments_mem_28593.mem + i_28529 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_28602 + i_28529 * 4, swap_term_mem_28592.mem + i_28529 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_27521 = res_27448 / res_27440;
    bool bounds_invalid_upwards_27522 = slt64(steps_27427, 1);
    bool valid_27523 = !bounds_invalid_upwards_27522;
    bool range_valid_c_27524;
    
    if (!valid_27523) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_27427,
                               " is invalid.",
                               "-> #0  cvadynmem.fut:70:56-67\n   #1  cvadynmem.fut:122:17-44\n   #2  cvadynmem.fut:111:1-166:19\n");
        if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_28651 = 4 * steps_27427;
    
    if (mem_28652_cached_sizze_28788 < (size_t) bytes_28651) {
        mem_28652 = realloc(mem_28652, bytes_28651);
        mem_28652_cached_sizze_28788 = bytes_28651;
    }
    for (int64_t i_28536 = 0; i_28536 < steps_27427; i_28536++) {
        int64_t index_primexp_28571 = add64(1, i_28536);
        float res_27528 = sitofp_i64_f32(index_primexp_28571);
        float res_27529 = res_27528 / sims_per_year_27521;
        
        ((float *) mem_28652)[i_28536] = res_27529;
    }
    
    bool bounds_invalid_upwards_27530 = slt64(paths_27426, 0);
    bool valid_27531 = !bounds_invalid_upwards_27530;
    bool range_valid_c_27532;
    
    if (!valid_27531) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_27426,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cvadynmem.fut:125:19-49\n   #4  cvadynmem.fut:111:1-166:19\n");
        if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
            return 1;
        return 1;
    }
    
    int64_t upper_bound_27535 = sub64(steps_27427, 1);
    float res_27536;
    
    res_27536 = futrts_sqrt32(dt_27449);
    
    int64_t binop_x_28666 = paths_27426 * steps_27427;
    int64_t bytes_28665 = 4 * binop_x_28666;
    
    if (mem_28667_cached_sizze_28789 < (size_t) bytes_28665) {
        mem_28667 = realloc(mem_28667, bytes_28665);
        mem_28667_cached_sizze_28789 = bytes_28665;
    }
    if (mem_28679_cached_sizze_28790 < (size_t) bytes_28651) {
        mem_28679 = realloc(mem_28679, bytes_28651);
        mem_28679_cached_sizze_28790 = bytes_28651;
    }
    if (mem_28693_cached_sizze_28791 < (size_t) bytes_28651) {
        mem_28693 = realloc(mem_28693, bytes_28651);
        mem_28693_cached_sizze_28791 = bytes_28651;
    }
    for (int64_t i_28772 = 0; i_28772 < steps_27427; i_28772++) {
        ((float *) mem_28693)[i_28772] = r0_27434;
    }
    for (int64_t i_28544 = 0; i_28544 < paths_27426; i_28544++) {
        int32_t res_27539 = sext_i64_i32(i_28544);
        int32_t x_27540 = lshr32(res_27539, 16);
        int32_t x_27541 = res_27539 ^ x_27540;
        int32_t x_27542 = mul32(73244475, x_27541);
        int32_t x_27543 = lshr32(x_27542, 16);
        int32_t x_27544 = x_27542 ^ x_27543;
        int32_t x_27545 = mul32(73244475, x_27544);
        int32_t x_27546 = lshr32(x_27545, 16);
        int32_t x_27547 = x_27545 ^ x_27546;
        int32_t unsign_arg_27548 = 777822902 ^ x_27547;
        int32_t unsign_arg_27549 = mul32(48271, unsign_arg_27548);
        int32_t unsign_arg_27550 = umod32(unsign_arg_27549, 2147483647);
        
        for (int64_t i_28540 = 0; i_28540 < steps_27427; i_28540++) {
            int32_t res_27553 = sext_i64_i32(i_28540);
            int32_t x_27554 = lshr32(res_27553, 16);
            int32_t x_27555 = res_27553 ^ x_27554;
            int32_t x_27556 = mul32(73244475, x_27555);
            int32_t x_27557 = lshr32(x_27556, 16);
            int32_t x_27558 = x_27556 ^ x_27557;
            int32_t x_27559 = mul32(73244475, x_27558);
            int32_t x_27560 = lshr32(x_27559, 16);
            int32_t x_27561 = x_27559 ^ x_27560;
            int32_t unsign_arg_27562 = unsign_arg_27550 ^ x_27561;
            int32_t unsign_arg_27563 = mul32(48271, unsign_arg_27562);
            int32_t unsign_arg_27564 = umod32(unsign_arg_27563, 2147483647);
            int32_t unsign_arg_27565 = mul32(48271, unsign_arg_27564);
            int32_t unsign_arg_27566 = umod32(unsign_arg_27565, 2147483647);
            float res_27567 = uitofp_i32_f32(unsign_arg_27564);
            float res_27568 = res_27567 / 2.1474836e9F;
            float res_27569 = uitofp_i32_f32(unsign_arg_27566);
            float res_27570 = res_27569 / 2.1474836e9F;
            float res_27571;
            
            res_27571 = futrts_log32(res_27568);
            
            float res_27572 = -2.0F * res_27571;
            float res_27573;
            
            res_27573 = futrts_sqrt32(res_27572);
            
            float res_27574 = 6.2831855F * res_27570;
            float res_27575;
            
            res_27575 = futrts_cos32(res_27574);
            
            float res_27576 = res_27573 * res_27575;
            
            ((float *) mem_28679)[i_28540] = res_27576;
        }
        memmove(mem_28667 + i_28544 * steps_27427 * 4, mem_28693 + 0,
                steps_27427 * (int32_t) sizeof(float));
        for (int64_t i_27579 = 0; i_27579 < upper_bound_27535; i_27579++) {
            bool y_27581 = slt64(i_27579, steps_27427);
            bool index_certs_27582;
            
            if (!y_27581) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27579,
                              "] out of bounds for array of shape [",
                              steps_27427, "].",
                              "-> #0  cvadynmem.fut:81:97-104\n   #1  cvadynmem.fut:133:32-62\n   #2  cvadynmem.fut:133:22-69\n   #3  cvadynmem.fut:111:1-166:19\n");
                if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
                    return 1;
                return 1;
            }
            
            float shortstep_arg_27583 = ((float *) mem_28679)[i_27579];
            float shortstep_arg_27584 = ((float *) mem_28667)[i_28544 *
                                                              steps_27427 +
                                                              i_27579];
            float y_27585 = b_27432 - shortstep_arg_27584;
            float x_27586 = a_27431 * y_27585;
            float x_27587 = dt_27449 * x_27586;
            float x_27588 = res_27536 * shortstep_arg_27583;
            float y_27589 = sigma_27433 * x_27588;
            float delta_r_27590 = x_27587 + y_27589;
            float res_27591 = shortstep_arg_27584 + delta_r_27590;
            int64_t i_27592 = add64(1, i_27579);
            bool x_27593 = sle64(0, i_27592);
            bool y_27594 = slt64(i_27592, steps_27427);
            bool bounds_check_27595 = x_27593 && y_27594;
            bool index_certs_27596;
            
            if (!bounds_check_27595) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_27592,
                              "] out of bounds for array of shape [",
                              steps_27427, "].",
                              "-> #0  cvadynmem.fut:81:58-105\n   #1  cvadynmem.fut:133:32-62\n   #2  cvadynmem.fut:133:22-69\n   #3  cvadynmem.fut:111:1-166:19\n");
                if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
                    return 1;
                return 1;
            }
            ((float *) mem_28667)[i_28544 * steps_27427 + i_27592] = res_27591;
        }
    }
    
    float res_27598 = sitofp_i64_f32(paths_27426);
    
    if (mem_28721_cached_sizze_28792 < (size_t) bytes_28651) {
        mem_28721 = realloc(mem_28721, bytes_28651);
        mem_28721_cached_sizze_28792 = bytes_28651;
    }
    
    bool loop_nonempty_28578 = slt64(0, paths_27426);
    float res_27720;
    float redout_28553 = 0.0F;
    
    for (int64_t i_28555 = 0; i_28555 < steps_27427; i_28555++) {
        int64_t index_primexp_28582 = add64(1, i_28555);
        float res_27728 = sitofp_i64_f32(index_primexp_28582);
        float res_27729 = res_27728 / sims_per_year_27521;
        float x_27735;
        
        if (loop_nonempty_28578) {
            float x_28579 = ((float *) mem_28652)[i_28555];
            
            x_27735 = x_28579;
        } else {
            x_27735 = 0.0F;
        }
        
        float res_27730;
        float redout_28550 = 0.0F;
        
        for (int64_t i_28551 = 0; i_28551 < paths_27426; i_28551++) {
            float x_27734 = ((float *) mem_28667)[i_28551 * steps_27427 +
                                                  i_28555];
            float res_27736;
            float redout_28548 = 0.0F;
            
            for (int64_t i_28549 = 0; i_28549 < n_27423; i_28549++) {
                float x_27740 = ((float *) mem_28596)[i_28549];
                float x_27741 = ((float *) mem_28598)[i_28549];
                int64_t x_27742 = ((int64_t *) mem_28600)[i_28549];
                float x_27743 = ((float *) mem_28602)[i_28549];
                int64_t i64_arg_27744 = sub64(x_27742, 1);
                float res_27745 = sitofp_i64_f32(i64_arg_27744);
                float y_27746 = x_27743 * res_27745;
                bool cond_27747 = y_27746 < x_27735;
                float ceil_arg_27748 = x_27735 / x_27743;
                float res_27749;
                
                res_27749 = futrts_ceil32(ceil_arg_27748);
                
                int64_t res_27750 = fptosi_f32_i64(res_27749);
                int64_t remaining_27751 = sub64(x_27742, res_27750);
                float res_27752;
                
                if (cond_27747) {
                    res_27752 = 0.0F;
                } else {
                    float nextpayment_27753 = x_27743 * res_27749;
                    bool bounds_invalid_upwards_27754 = slt64(remaining_27751,
                                                              1);
                    bool valid_27755 = !bounds_invalid_upwards_27754;
                    bool range_valid_c_27756;
                    
                    if (!valid_27755) {
                        ctx->error =
                            msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                                      "Range ", 1, "..", 2, "...",
                                      remaining_27751, " is invalid.",
                                      "-> #0  cvadynmem.fut:40:30-45\n   #1  cvadynmem.fut:48:27-71\n   #2  cvadynmem.fut:140:51-76\n   #3  cvadynmem.fut:138:35-140:84\n   #4  /prelude/soacs.fut:56:19-23\n   #5  /prelude/soacs.fut:56:3-37\n   #6  cvadynmem.fut:137:25-144:33\n   #7  cvadynmem.fut:136:21-144:45\n   #8  cvadynmem.fut:111:1-166:19\n");
                        if (memblock_unref(ctx, &out_mem_28763,
                                           "out_mem_28763") != 0)
                            return 1;
                        return 1;
                    }
                    
                    float y_27758 = nextpayment_27753 - x_27735;
                    float negate_arg_27759 = a_27431 * y_27758;
                    float exp_arg_27760 = 0.0F - negate_arg_27759;
                    float res_27761 = fpow32(2.7182817F, exp_arg_27760);
                    float x_27762 = 1.0F - res_27761;
                    float B_27763 = x_27762 / a_27431;
                    float x_27764 = B_27763 - nextpayment_27753;
                    float x_27765 = x_27735 + x_27764;
                    float x_27771 = y_27455 * x_27765;
                    float A1_27772 = x_27771 / x_27451;
                    float y_27773 = fpow32(B_27763, 2.0F);
                    float x_27774 = x_27453 * y_27773;
                    float A2_27776 = x_27774 / y_27456;
                    float exp_arg_27777 = A1_27772 - A2_27776;
                    float res_27778 = fpow32(2.7182817F, exp_arg_27777);
                    float negate_arg_27779 = x_27734 * B_27763;
                    float exp_arg_27780 = 0.0F - negate_arg_27779;
                    float res_27781 = fpow32(2.7182817F, exp_arg_27780);
                    float res_27782 = res_27778 * res_27781;
                    bool y_27783 = slt64(0, remaining_27751);
                    bool index_certs_27784;
                    
                    if (!y_27783) {
                        ctx->error =
                            msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                      "Index [", 0,
                                      "] out of bounds for array of shape [",
                                      remaining_27751, "].",
                                      "-> #0  cvadynmem.fut:50:38-63\n   #1  cvadynmem.fut:140:51-76\n   #2  cvadynmem.fut:138:35-140:84\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cvadynmem.fut:137:25-144:33\n   #6  cvadynmem.fut:136:21-144:45\n   #7  cvadynmem.fut:111:1-166:19\n");
                        if (memblock_unref(ctx, &out_mem_28763,
                                           "out_mem_28763") != 0)
                            return 1;
                        return 1;
                    }
                    
                    float binop_y_27785 = sitofp_i64_f32(remaining_27751);
                    float binop_y_27786 = x_27743 * binop_y_27785;
                    float index_primexp_27787 = nextpayment_27753 +
                          binop_y_27786;
                    float y_27788 = index_primexp_27787 - x_27735;
                    float negate_arg_27789 = a_27431 * y_27788;
                    float exp_arg_27790 = 0.0F - negate_arg_27789;
                    float res_27791 = fpow32(2.7182817F, exp_arg_27790);
                    float x_27792 = 1.0F - res_27791;
                    float B_27793 = x_27792 / a_27431;
                    float x_27794 = B_27793 - index_primexp_27787;
                    float x_27795 = x_27735 + x_27794;
                    float x_27796 = y_27455 * x_27795;
                    float A1_27797 = x_27796 / x_27451;
                    float y_27798 = fpow32(B_27793, 2.0F);
                    float x_27799 = x_27453 * y_27798;
                    float A2_27800 = x_27799 / y_27456;
                    float exp_arg_27801 = A1_27797 - A2_27800;
                    float res_27802 = fpow32(2.7182817F, exp_arg_27801);
                    float negate_arg_27803 = x_27734 * B_27793;
                    float exp_arg_27804 = 0.0F - negate_arg_27803;
                    float res_27805 = fpow32(2.7182817F, exp_arg_27804);
                    float res_27806 = res_27802 * res_27805;
                    float res_27807;
                    float redout_28546 = 0.0F;
                    
                    for (int64_t i_28547 = 0; i_28547 < remaining_27751;
                         i_28547++) {
                        int64_t index_primexp_28577 = add64(1, i_28547);
                        float res_27812 = sitofp_i64_f32(index_primexp_28577);
                        float res_27813 = x_27743 * res_27812;
                        float res_27814 = nextpayment_27753 + res_27813;
                        float y_27815 = res_27814 - x_27735;
                        float negate_arg_27816 = a_27431 * y_27815;
                        float exp_arg_27817 = 0.0F - negate_arg_27816;
                        float res_27818 = fpow32(2.7182817F, exp_arg_27817);
                        float x_27819 = 1.0F - res_27818;
                        float B_27820 = x_27819 / a_27431;
                        float x_27821 = B_27820 - res_27814;
                        float x_27822 = x_27735 + x_27821;
                        float x_27823 = y_27455 * x_27822;
                        float A1_27824 = x_27823 / x_27451;
                        float y_27825 = fpow32(B_27820, 2.0F);
                        float x_27826 = x_27453 * y_27825;
                        float A2_27827 = x_27826 / y_27456;
                        float exp_arg_27828 = A1_27824 - A2_27827;
                        float res_27829 = fpow32(2.7182817F, exp_arg_27828);
                        float negate_arg_27830 = x_27734 * B_27820;
                        float exp_arg_27831 = 0.0F - negate_arg_27830;
                        float res_27832 = fpow32(2.7182817F, exp_arg_27831);
                        float res_27833 = res_27829 * res_27832;
                        float res_27810 = res_27833 + redout_28546;
                        float redout_tmp_28780 = res_27810;
                        
                        redout_28546 = redout_tmp_28780;
                    }
                    res_27807 = redout_28546;
                    
                    float x_27834 = res_27782 - res_27806;
                    float x_27835 = x_27740 * x_27743;
                    float y_27836 = res_27807 * x_27835;
                    float y_27837 = x_27834 - y_27836;
                    float res_27838 = x_27741 * y_27837;
                    float res_27839 = x_27741 * res_27838;
                    
                    res_27752 = res_27839;
                }
                
                float res_27739 = res_27752 + redout_28548;
                float redout_tmp_28779 = res_27739;
                
                redout_28548 = redout_tmp_28779;
            }
            res_27736 = redout_28548;
            
            float res_27840 = fmax32(0.0F, res_27736);
            float res_27733 = res_27840 + redout_28550;
            float redout_tmp_28778 = res_27733;
            
            redout_28550 = redout_tmp_28778;
        }
        res_27730 = redout_28550;
        
        float res_27841 = res_27730 / res_27598;
        float negate_arg_27842 = a_27431 * res_27729;
        float exp_arg_27843 = 0.0F - negate_arg_27842;
        float res_27844 = fpow32(2.7182817F, exp_arg_27843);
        float x_27845 = 1.0F - res_27844;
        float B_27846 = x_27845 / a_27431;
        float x_27847 = B_27846 - res_27729;
        float x_27848 = y_27455 * x_27847;
        float A1_27849 = x_27848 / x_27451;
        float y_27850 = fpow32(B_27846, 2.0F);
        float x_27851 = x_27453 * y_27850;
        float A2_27852 = x_27851 / y_27456;
        float exp_arg_27853 = A1_27849 - A2_27852;
        float res_27854 = fpow32(2.7182817F, exp_arg_27853);
        float negate_arg_27855 = 5.0e-2F * B_27846;
        float exp_arg_27856 = 0.0F - negate_arg_27855;
        float res_27857 = fpow32(2.7182817F, exp_arg_27856);
        float res_27858 = res_27854 * res_27857;
        float res_27859 = res_27841 * res_27858;
        float res_27724 = res_27859 + redout_28553;
        
        ((float *) mem_28721)[i_28555] = res_27841;
        
        float redout_tmp_28776 = res_27724;
        
        redout_28553 = redout_tmp_28776;
    }
    res_27720 = redout_28553;
    
    float CVA_27862 = 6.0e-3F * res_27720;
    struct memblock mem_28735;
    
    mem_28735.references = NULL;
    if (memblock_alloc(ctx, &mem_28735, bytes_28651, "mem_28735")) {
        err = 1;
        goto cleanup;
    }
    memmove(mem_28735.mem + 0, mem_28721 + 0, steps_27427 *
            (int32_t) sizeof(float));
    out_arrsizze_28764 = steps_27427;
    if (memblock_set(ctx, &out_mem_28763, &mem_28735, "mem_28735") != 0)
        return 1;
    scalar_out_28762 = CVA_27862;
    *out_scalar_out_28781 = scalar_out_28762;
    (*out_mem_p_28782).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_28782, &out_mem_28763, "out_mem_28763") !=
        0)
        return 1;
    *out_out_arrsizze_28783 = out_arrsizze_28764;
    if (memblock_unref(ctx, &mem_28735, "mem_28735") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_28763, "out_mem_28763") != 0)
        return 1;
    
  cleanup:
    { }
    free(mem_28596);
    free(mem_28598);
    free(mem_28600);
    free(mem_28602);
    free(mem_28652);
    free(mem_28667);
    free(mem_28679);
    free(mem_28693);
    free(mem_28721);
    return err;
}
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_28793,
                       int64_t paths_27863, int64_t steps_27864)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_28593_cached_sizze_28794 = 0;
    char *mem_28593 = NULL;
    size_t mem_28595_cached_sizze_28795 = 0;
    char *mem_28595 = NULL;
    size_t mem_28597_cached_sizze_28796 = 0;
    char *mem_28597 = NULL;
    size_t mem_28599_cached_sizze_28797 = 0;
    char *mem_28599 = NULL;
    size_t mem_28601_cached_sizze_28798 = 0;
    char *mem_28601 = NULL;
    size_t mem_28603_cached_sizze_28799 = 0;
    char *mem_28603 = NULL;
    size_t mem_28605_cached_sizze_28800 = 0;
    char *mem_28605 = NULL;
    size_t mem_28663_cached_sizze_28801 = 0;
    char *mem_28663 = NULL;
    size_t mem_28678_cached_sizze_28802 = 0;
    char *mem_28678 = NULL;
    size_t mem_28690_cached_sizze_28803 = 0;
    char *mem_28690 = NULL;
    size_t mem_28704_cached_sizze_28804 = 0;
    char *mem_28704 = NULL;
    float scalar_out_28762;
    
    if (mem_28593_cached_sizze_28794 < (size_t) 180) {
        mem_28593 = realloc(mem_28593, 180);
        mem_28593_cached_sizze_28794 = 180;
    }
    
    struct memblock testzistatic_array_28763 = ctx->testzistatic_array_28763;
    
    memmove(mem_28593 + 0, testzistatic_array_28763.mem + 0, 45 *
            (int32_t) sizeof(float));
    if (mem_28595_cached_sizze_28795 < (size_t) 360) {
        mem_28595 = realloc(mem_28595, 360);
        mem_28595_cached_sizze_28795 = 360;
    }
    
    struct memblock testzistatic_array_28764 = ctx->testzistatic_array_28764;
    
    memmove(mem_28595 + 0, testzistatic_array_28764.mem + 0, 45 *
            (int32_t) sizeof(int64_t));
    if (mem_28597_cached_sizze_28796 < (size_t) 180) {
        mem_28597 = realloc(mem_28597, 180);
        mem_28597_cached_sizze_28796 = 180;
    }
    
    struct memblock testzistatic_array_28765 = ctx->testzistatic_array_28765;
    
    memmove(mem_28597 + 0, testzistatic_array_28765.mem + 0, 45 *
            (int32_t) sizeof(float));
    
    float res_27868;
    float redout_28517 = -INFINITY;
    
    for (int32_t i_28563 = 0; i_28563 < 45; i_28563++) {
        int64_t i_28518 = sext_i32_i64(i_28563);
        float x_27872 = ((float *) mem_28597)[i_28518];
        int64_t x_27873 = ((int64_t *) mem_28595)[i_28518];
        float res_27874 = sitofp_i64_f32(x_27873);
        float res_27875 = x_27872 * res_27874;
        float res_27871 = fmax32(res_27875, redout_28517);
        float redout_tmp_28766 = res_27871;
        
        redout_28517 = redout_tmp_28766;
    }
    res_27868 = redout_28517;
    
    float res_27876 = sitofp_i64_f32(steps_27864);
    float dt_27877 = res_27868 / res_27876;
    
    if (mem_28599_cached_sizze_28797 < (size_t) 180) {
        mem_28599 = realloc(mem_28599, 180);
        mem_28599_cached_sizze_28797 = 180;
    }
    if (mem_28601_cached_sizze_28798 < (size_t) 180) {
        mem_28601 = realloc(mem_28601, 180);
        mem_28601_cached_sizze_28798 = 180;
    }
    if (mem_28603_cached_sizze_28799 < (size_t) 360) {
        mem_28603 = realloc(mem_28603, 360);
        mem_28603_cached_sizze_28799 = 360;
    }
    if (mem_28605_cached_sizze_28800 < (size_t) 180) {
        mem_28605 = realloc(mem_28605, 180);
        mem_28605_cached_sizze_28800 = 180;
    }
    for (int32_t i_28571 = 0; i_28571 < 45; i_28571++) {
        int64_t i_28529 = sext_i32_i64(i_28571);
        bool x_27884 = sle64(0, i_28529);
        bool y_27885 = slt64(i_28529, 45);
        bool bounds_check_27886 = x_27884 && y_27885;
        bool index_certs_27887;
        
        if (!bounds_check_27886) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", i_28529,
                                   "] out of bounds for array of shape [", 45,
                                   "].",
                                   "-> #0  cvadynmem.fut:117:15-26\n   #1  cvadynmem.fut:116:17-120:85\n   #2  cvadynmem.fut:175:3-177:129\n   #3  cvadynmem.fut:174:1-177:137\n");
            return 1;
        }
        
        float res_27888 = ((float *) mem_28597)[i_28529];
        int64_t res_27889 = ((int64_t *) mem_28595)[i_28529];
        bool bounds_invalid_upwards_27891 = slt64(res_27889, 1);
        bool valid_27892 = !bounds_invalid_upwards_27891;
        bool range_valid_c_27893;
        
        if (!valid_27892) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_27889,
                          " is invalid.",
                          "-> #0  cvadynmem.fut:64:29-48\n   #1  cvadynmem.fut:105:25-65\n   #2  cvadynmem.fut:120:16-62\n   #3  cvadynmem.fut:116:17-120:85\n   #4  cvadynmem.fut:175:3-177:129\n   #5  cvadynmem.fut:174:1-177:137\n");
            return 1;
        }
        
        bool y_27895 = slt64(0, res_27889);
        bool index_certs_27896;
        
        if (!y_27895) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_27889, "].",
                                   "-> #0  cvadynmem.fut:106:47-70\n   #1  cvadynmem.fut:120:16-62\n   #2  cvadynmem.fut:116:17-120:85\n   #3  cvadynmem.fut:175:3-177:129\n   #4  cvadynmem.fut:174:1-177:137\n");
            return 1;
        }
        
        float binop_y_27897 = sitofp_i64_f32(res_27889);
        float index_primexp_27898 = res_27888 * binop_y_27897;
        float negate_arg_27899 = 1.0e-2F * index_primexp_27898;
        float exp_arg_27900 = 0.0F - negate_arg_27899;
        float res_27901 = fpow32(2.7182817F, exp_arg_27900);
        float x_27902 = 1.0F - res_27901;
        float B_27903 = x_27902 / 1.0e-2F;
        float x_27904 = B_27903 - index_primexp_27898;
        float x_27905 = 4.4999997e-6F * x_27904;
        float A1_27906 = x_27905 / 1.0e-4F;
        float y_27907 = fpow32(B_27903, 2.0F);
        float x_27908 = 1.0000001e-6F * y_27907;
        float A2_27909 = x_27908 / 4.0e-2F;
        float exp_arg_27910 = A1_27906 - A2_27909;
        float res_27911 = fpow32(2.7182817F, exp_arg_27910);
        float negate_arg_27912 = 5.0e-2F * B_27903;
        float exp_arg_27913 = 0.0F - negate_arg_27912;
        float res_27914 = fpow32(2.7182817F, exp_arg_27913);
        float res_27915 = res_27911 * res_27914;
        float res_27916;
        float redout_28519 = 0.0F;
        
        for (int64_t i_28520 = 0; i_28520 < res_27889; i_28520++) {
            int64_t index_primexp_28565 = add64(1, i_28520);
            float res_27921 = sitofp_i64_f32(index_primexp_28565);
            float res_27922 = res_27888 * res_27921;
            float negate_arg_27923 = 1.0e-2F * res_27922;
            float exp_arg_27924 = 0.0F - negate_arg_27923;
            float res_27925 = fpow32(2.7182817F, exp_arg_27924);
            float x_27926 = 1.0F - res_27925;
            float B_27927 = x_27926 / 1.0e-2F;
            float x_27928 = B_27927 - res_27922;
            float x_27929 = 4.4999997e-6F * x_27928;
            float A1_27930 = x_27929 / 1.0e-4F;
            float y_27931 = fpow32(B_27927, 2.0F);
            float x_27932 = 1.0000001e-6F * y_27931;
            float A2_27933 = x_27932 / 4.0e-2F;
            float exp_arg_27934 = A1_27930 - A2_27933;
            float res_27935 = fpow32(2.7182817F, exp_arg_27934);
            float negate_arg_27936 = 5.0e-2F * B_27927;
            float exp_arg_27937 = 0.0F - negate_arg_27936;
            float res_27938 = fpow32(2.7182817F, exp_arg_27937);
            float res_27939 = res_27935 * res_27938;
            float res_27919 = res_27939 + redout_28519;
            float redout_tmp_28771 = res_27919;
            
            redout_28519 = redout_tmp_28771;
        }
        res_27916 = redout_28519;
        
        float x_27940 = 1.0F - res_27915;
        float y_27941 = res_27888 * res_27916;
        float res_27942 = x_27940 / y_27941;
        
        ((float *) mem_28599)[i_28529] = res_27942;
        memmove(mem_28601 + i_28529 * 4, mem_28593 + i_28529 * 4,
                (int32_t) sizeof(float));
        memmove(mem_28603 + i_28529 * 8, mem_28595 + i_28529 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_28605 + i_28529 * 4, mem_28597 + i_28529 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_27943 = res_27876 / res_27868;
    bool bounds_invalid_upwards_27944 = slt64(steps_27864, 1);
    bool valid_27945 = !bounds_invalid_upwards_27944;
    bool range_valid_c_27946;
    
    if (!valid_27945) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_27864,
                               " is invalid.",
                               "-> #0  cvadynmem.fut:70:56-67\n   #1  cvadynmem.fut:122:17-44\n   #2  cvadynmem.fut:175:3-177:129\n   #3  cvadynmem.fut:174:1-177:137\n");
        return 1;
    }
    
    int64_t bytes_28662 = 4 * steps_27864;
    
    if (mem_28663_cached_sizze_28801 < (size_t) bytes_28662) {
        mem_28663 = realloc(mem_28663, bytes_28662);
        mem_28663_cached_sizze_28801 = bytes_28662;
    }
    for (int64_t i_28536 = 0; i_28536 < steps_27864; i_28536++) {
        int64_t index_primexp_28573 = add64(1, i_28536);
        float res_27950 = sitofp_i64_f32(index_primexp_28573);
        float res_27951 = res_27950 / sims_per_year_27943;
        
        ((float *) mem_28663)[i_28536] = res_27951;
    }
    
    bool bounds_invalid_upwards_27952 = slt64(paths_27863, 0);
    bool valid_27953 = !bounds_invalid_upwards_27952;
    bool range_valid_c_27954;
    
    if (!valid_27953) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_27863,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cvadynmem.fut:125:19-49\n   #4  cvadynmem.fut:175:3-177:129\n   #5  cvadynmem.fut:174:1-177:137\n");
        return 1;
    }
    
    int64_t upper_bound_27957 = sub64(steps_27864, 1);
    float res_27958;
    
    res_27958 = futrts_sqrt32(dt_27877);
    
    int64_t binop_x_28677 = paths_27863 * steps_27864;
    int64_t bytes_28676 = 4 * binop_x_28677;
    
    if (mem_28678_cached_sizze_28802 < (size_t) bytes_28676) {
        mem_28678 = realloc(mem_28678, bytes_28676);
        mem_28678_cached_sizze_28802 = bytes_28676;
    }
    if (mem_28690_cached_sizze_28803 < (size_t) bytes_28662) {
        mem_28690 = realloc(mem_28690, bytes_28662);
        mem_28690_cached_sizze_28803 = bytes_28662;
    }
    if (mem_28704_cached_sizze_28804 < (size_t) bytes_28662) {
        mem_28704 = realloc(mem_28704, bytes_28662);
        mem_28704_cached_sizze_28804 = bytes_28662;
    }
    for (int64_t i_28773 = 0; i_28773 < steps_27864; i_28773++) {
        ((float *) mem_28704)[i_28773] = 5.0e-2F;
    }
    for (int64_t i_28544 = 0; i_28544 < paths_27863; i_28544++) {
        int32_t res_27961 = sext_i64_i32(i_28544);
        int32_t x_27962 = lshr32(res_27961, 16);
        int32_t x_27963 = res_27961 ^ x_27962;
        int32_t x_27964 = mul32(73244475, x_27963);
        int32_t x_27965 = lshr32(x_27964, 16);
        int32_t x_27966 = x_27964 ^ x_27965;
        int32_t x_27967 = mul32(73244475, x_27966);
        int32_t x_27968 = lshr32(x_27967, 16);
        int32_t x_27969 = x_27967 ^ x_27968;
        int32_t unsign_arg_27970 = 777822902 ^ x_27969;
        int32_t unsign_arg_27971 = mul32(48271, unsign_arg_27970);
        int32_t unsign_arg_27972 = umod32(unsign_arg_27971, 2147483647);
        
        for (int64_t i_28540 = 0; i_28540 < steps_27864; i_28540++) {
            int32_t res_27975 = sext_i64_i32(i_28540);
            int32_t x_27976 = lshr32(res_27975, 16);
            int32_t x_27977 = res_27975 ^ x_27976;
            int32_t x_27978 = mul32(73244475, x_27977);
            int32_t x_27979 = lshr32(x_27978, 16);
            int32_t x_27980 = x_27978 ^ x_27979;
            int32_t x_27981 = mul32(73244475, x_27980);
            int32_t x_27982 = lshr32(x_27981, 16);
            int32_t x_27983 = x_27981 ^ x_27982;
            int32_t unsign_arg_27984 = unsign_arg_27972 ^ x_27983;
            int32_t unsign_arg_27985 = mul32(48271, unsign_arg_27984);
            int32_t unsign_arg_27986 = umod32(unsign_arg_27985, 2147483647);
            int32_t unsign_arg_27987 = mul32(48271, unsign_arg_27986);
            int32_t unsign_arg_27988 = umod32(unsign_arg_27987, 2147483647);
            float res_27989 = uitofp_i32_f32(unsign_arg_27986);
            float res_27990 = res_27989 / 2.1474836e9F;
            float res_27991 = uitofp_i32_f32(unsign_arg_27988);
            float res_27992 = res_27991 / 2.1474836e9F;
            float res_27993;
            
            res_27993 = futrts_log32(res_27990);
            
            float res_27994 = -2.0F * res_27993;
            float res_27995;
            
            res_27995 = futrts_sqrt32(res_27994);
            
            float res_27996 = 6.2831855F * res_27992;
            float res_27997;
            
            res_27997 = futrts_cos32(res_27996);
            
            float res_27998 = res_27995 * res_27997;
            
            ((float *) mem_28690)[i_28540] = res_27998;
        }
        memmove(mem_28678 + i_28544 * steps_27864 * 4, mem_28704 + 0,
                steps_27864 * (int32_t) sizeof(float));
        for (int64_t i_28001 = 0; i_28001 < upper_bound_27957; i_28001++) {
            bool y_28003 = slt64(i_28001, steps_27864);
            bool index_certs_28004;
            
            if (!y_28003) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_28001,
                              "] out of bounds for array of shape [",
                              steps_27864, "].",
                              "-> #0  cvadynmem.fut:81:97-104\n   #1  cvadynmem.fut:133:32-62\n   #2  cvadynmem.fut:133:22-69\n   #3  cvadynmem.fut:175:3-177:129\n   #4  cvadynmem.fut:174:1-177:137\n");
                return 1;
            }
            
            float shortstep_arg_28005 = ((float *) mem_28690)[i_28001];
            float shortstep_arg_28006 = ((float *) mem_28678)[i_28544 *
                                                              steps_27864 +
                                                              i_28001];
            float y_28007 = 5.0e-2F - shortstep_arg_28006;
            float x_28008 = 1.0e-2F * y_28007;
            float x_28009 = dt_27877 * x_28008;
            float x_28010 = res_27958 * shortstep_arg_28005;
            float y_28011 = 1.0e-3F * x_28010;
            float delta_r_28012 = x_28009 + y_28011;
            float res_28013 = shortstep_arg_28006 + delta_r_28012;
            int64_t i_28014 = add64(1, i_28001);
            bool x_28015 = sle64(0, i_28014);
            bool y_28016 = slt64(i_28014, steps_27864);
            bool bounds_check_28017 = x_28015 && y_28016;
            bool index_certs_28018;
            
            if (!bounds_check_28017) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_28014,
                              "] out of bounds for array of shape [",
                              steps_27864, "].",
                              "-> #0  cvadynmem.fut:81:58-105\n   #1  cvadynmem.fut:133:32-62\n   #2  cvadynmem.fut:133:22-69\n   #3  cvadynmem.fut:175:3-177:129\n   #4  cvadynmem.fut:174:1-177:137\n");
                return 1;
            }
            ((float *) mem_28678)[i_28544 * steps_27864 + i_28014] = res_28013;
        }
    }
    
    float res_28020 = sitofp_i64_f32(paths_27863);
    bool loop_nonempty_28581 = slt64(0, paths_27863);
    float res_28027;
    float redout_28552 = 0.0F;
    
    for (int64_t i_28553 = 0; i_28553 < steps_27864; i_28553++) {
        int64_t index_primexp_28585 = add64(1, i_28553);
        float res_28034 = sitofp_i64_f32(index_primexp_28585);
        float res_28035 = res_28034 / sims_per_year_27943;
        float x_28041;
        
        if (loop_nonempty_28581) {
            float x_28582 = ((float *) mem_28663)[i_28553];
            
            x_28041 = x_28582;
        } else {
            x_28041 = 0.0F;
        }
        
        float res_28036;
        float redout_28550 = 0.0F;
        
        for (int64_t i_28551 = 0; i_28551 < paths_27863; i_28551++) {
            float x_28040 = ((float *) mem_28678)[i_28551 * steps_27864 +
                                                  i_28553];
            float res_28042;
            float redout_28548 = 0.0F;
            
            for (int32_t i_28580 = 0; i_28580 < 45; i_28580++) {
                int64_t i_28549 = sext_i32_i64(i_28580);
                float x_28046 = ((float *) mem_28599)[i_28549];
                float x_28047 = ((float *) mem_28601)[i_28549];
                int64_t x_28048 = ((int64_t *) mem_28603)[i_28549];
                float x_28049 = ((float *) mem_28605)[i_28549];
                int64_t i64_arg_28050 = sub64(x_28048, 1);
                float res_28051 = sitofp_i64_f32(i64_arg_28050);
                float y_28052 = x_28049 * res_28051;
                bool cond_28053 = y_28052 < x_28041;
                float ceil_arg_28054 = x_28041 / x_28049;
                float res_28055;
                
                res_28055 = futrts_ceil32(ceil_arg_28054);
                
                int64_t res_28056 = fptosi_f32_i64(res_28055);
                int64_t remaining_28057 = sub64(x_28048, res_28056);
                float res_28058;
                
                if (cond_28053) {
                    res_28058 = 0.0F;
                } else {
                    float nextpayment_28059 = x_28049 * res_28055;
                    bool bounds_invalid_upwards_28060 = slt64(remaining_28057,
                                                              1);
                    bool valid_28061 = !bounds_invalid_upwards_28060;
                    bool range_valid_c_28062;
                    
                    if (!valid_28061) {
                        ctx->error =
                            msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                                      "Range ", 1, "..", 2, "...",
                                      remaining_28057, " is invalid.",
                                      "-> #0  cvadynmem.fut:40:30-45\n   #1  cvadynmem.fut:48:27-71\n   #2  cvadynmem.fut:140:51-76\n   #3  cvadynmem.fut:138:35-140:84\n   #4  /prelude/soacs.fut:56:19-23\n   #5  /prelude/soacs.fut:56:3-37\n   #6  cvadynmem.fut:137:25-144:33\n   #7  cvadynmem.fut:136:21-144:45\n   #8  cvadynmem.fut:175:3-177:129\n   #9  cvadynmem.fut:174:1-177:137\n");
                        return 1;
                    }
                    
                    float y_28064 = nextpayment_28059 - x_28041;
                    float negate_arg_28065 = 1.0e-2F * y_28064;
                    float exp_arg_28066 = 0.0F - negate_arg_28065;
                    float res_28067 = fpow32(2.7182817F, exp_arg_28066);
                    float x_28068 = 1.0F - res_28067;
                    float B_28069 = x_28068 / 1.0e-2F;
                    float x_28070 = B_28069 - nextpayment_28059;
                    float x_28071 = x_28041 + x_28070;
                    float x_28072 = 4.4999997e-6F * x_28071;
                    float A1_28073 = x_28072 / 1.0e-4F;
                    float y_28074 = fpow32(B_28069, 2.0F);
                    float x_28075 = 1.0000001e-6F * y_28074;
                    float A2_28076 = x_28075 / 4.0e-2F;
                    float exp_arg_28077 = A1_28073 - A2_28076;
                    float res_28078 = fpow32(2.7182817F, exp_arg_28077);
                    float negate_arg_28079 = x_28040 * B_28069;
                    float exp_arg_28080 = 0.0F - negate_arg_28079;
                    float res_28081 = fpow32(2.7182817F, exp_arg_28080);
                    float res_28082 = res_28078 * res_28081;
                    bool y_28083 = slt64(0, remaining_28057);
                    bool index_certs_28084;
                    
                    if (!y_28083) {
                        ctx->error =
                            msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                      "Index [", 0,
                                      "] out of bounds for array of shape [",
                                      remaining_28057, "].",
                                      "-> #0  cvadynmem.fut:50:38-63\n   #1  cvadynmem.fut:140:51-76\n   #2  cvadynmem.fut:138:35-140:84\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cvadynmem.fut:137:25-144:33\n   #6  cvadynmem.fut:136:21-144:45\n   #7  cvadynmem.fut:175:3-177:129\n   #8  cvadynmem.fut:174:1-177:137\n");
                        return 1;
                    }
                    
                    float binop_y_28085 = sitofp_i64_f32(remaining_28057);
                    float binop_y_28086 = x_28049 * binop_y_28085;
                    float index_primexp_28087 = nextpayment_28059 +
                          binop_y_28086;
                    float y_28088 = index_primexp_28087 - x_28041;
                    float negate_arg_28089 = 1.0e-2F * y_28088;
                    float exp_arg_28090 = 0.0F - negate_arg_28089;
                    float res_28091 = fpow32(2.7182817F, exp_arg_28090);
                    float x_28092 = 1.0F - res_28091;
                    float B_28093 = x_28092 / 1.0e-2F;
                    float x_28094 = B_28093 - index_primexp_28087;
                    float x_28095 = x_28041 + x_28094;
                    float x_28096 = 4.4999997e-6F * x_28095;
                    float A1_28097 = x_28096 / 1.0e-4F;
                    float y_28098 = fpow32(B_28093, 2.0F);
                    float x_28099 = 1.0000001e-6F * y_28098;
                    float A2_28100 = x_28099 / 4.0e-2F;
                    float exp_arg_28101 = A1_28097 - A2_28100;
                    float res_28102 = fpow32(2.7182817F, exp_arg_28101);
                    float negate_arg_28103 = x_28040 * B_28093;
                    float exp_arg_28104 = 0.0F - negate_arg_28103;
                    float res_28105 = fpow32(2.7182817F, exp_arg_28104);
                    float res_28106 = res_28102 * res_28105;
                    float res_28107;
                    float redout_28546 = 0.0F;
                    
                    for (int64_t i_28547 = 0; i_28547 < remaining_28057;
                         i_28547++) {
                        int64_t index_primexp_28579 = add64(1, i_28547);
                        float res_28112 = sitofp_i64_f32(index_primexp_28579);
                        float res_28113 = x_28049 * res_28112;
                        float res_28114 = nextpayment_28059 + res_28113;
                        float y_28115 = res_28114 - x_28041;
                        float negate_arg_28116 = 1.0e-2F * y_28115;
                        float exp_arg_28117 = 0.0F - negate_arg_28116;
                        float res_28118 = fpow32(2.7182817F, exp_arg_28117);
                        float x_28119 = 1.0F - res_28118;
                        float B_28120 = x_28119 / 1.0e-2F;
                        float x_28121 = B_28120 - res_28114;
                        float x_28122 = x_28041 + x_28121;
                        float x_28123 = 4.4999997e-6F * x_28122;
                        float A1_28124 = x_28123 / 1.0e-4F;
                        float y_28125 = fpow32(B_28120, 2.0F);
                        float x_28126 = 1.0000001e-6F * y_28125;
                        float A2_28127 = x_28126 / 4.0e-2F;
                        float exp_arg_28128 = A1_28124 - A2_28127;
                        float res_28129 = fpow32(2.7182817F, exp_arg_28128);
                        float negate_arg_28130 = x_28040 * B_28120;
                        float exp_arg_28131 = 0.0F - negate_arg_28130;
                        float res_28132 = fpow32(2.7182817F, exp_arg_28131);
                        float res_28133 = res_28129 * res_28132;
                        float res_28110 = res_28133 + redout_28546;
                        float redout_tmp_28780 = res_28110;
                        
                        redout_28546 = redout_tmp_28780;
                    }
                    res_28107 = redout_28546;
                    
                    float x_28134 = res_28082 - res_28106;
                    float x_28135 = x_28046 * x_28049;
                    float y_28136 = res_28107 * x_28135;
                    float y_28137 = x_28134 - y_28136;
                    float res_28138 = x_28047 * y_28137;
                    float res_28139 = x_28047 * res_28138;
                    
                    res_28058 = res_28139;
                }
                
                float res_28045 = res_28058 + redout_28548;
                float redout_tmp_28779 = res_28045;
                
                redout_28548 = redout_tmp_28779;
            }
            res_28042 = redout_28548;
            
            float res_28140 = fmax32(0.0F, res_28042);
            float res_28039 = res_28140 + redout_28550;
            float redout_tmp_28778 = res_28039;
            
            redout_28550 = redout_tmp_28778;
        }
        res_28036 = redout_28550;
        
        float res_28141 = res_28036 / res_28020;
        float negate_arg_28142 = 1.0e-2F * res_28035;
        float exp_arg_28143 = 0.0F - negate_arg_28142;
        float res_28144 = fpow32(2.7182817F, exp_arg_28143);
        float x_28145 = 1.0F - res_28144;
        float B_28146 = x_28145 / 1.0e-2F;
        float x_28147 = B_28146 - res_28035;
        float x_28148 = 4.4999997e-6F * x_28147;
        float A1_28149 = x_28148 / 1.0e-4F;
        float y_28150 = fpow32(B_28146, 2.0F);
        float x_28151 = 1.0000001e-6F * y_28150;
        float A2_28152 = x_28151 / 4.0e-2F;
        float exp_arg_28153 = A1_28149 - A2_28152;
        float res_28154 = fpow32(2.7182817F, exp_arg_28153);
        float negate_arg_28155 = 5.0e-2F * B_28146;
        float exp_arg_28156 = 0.0F - negate_arg_28155;
        float res_28157 = fpow32(2.7182817F, exp_arg_28156);
        float res_28158 = res_28154 * res_28157;
        float res_28159 = res_28141 * res_28158;
        float res_28030 = res_28159 + redout_28552;
        float redout_tmp_28777 = res_28030;
        
        redout_28552 = redout_tmp_28777;
    }
    res_28027 = redout_28552;
    
    float CVA_28161 = 6.0e-3F * res_28027;
    
    scalar_out_28762 = CVA_28161;
    *out_scalar_out_28793 = scalar_out_28762;
    
  cleanup:
    { }
    free(mem_28593);
    free(mem_28595);
    free(mem_28597);
    free(mem_28599);
    free(mem_28601);
    free(mem_28603);
    free(mem_28605);
    free(mem_28663);
    free(mem_28678);
    free(mem_28690);
    free(mem_28704);
    return err;
}
static int futrts_test2(struct futhark_context *ctx,
                        float *out_scalar_out_28808, int64_t paths_28162,
                        int64_t steps_28163, int64_t numswaps_28164)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_28593_cached_sizze_28809 = 0;
    char *mem_28593 = NULL;
    size_t mem_28595_cached_sizze_28810 = 0;
    char *mem_28595 = NULL;
    size_t mem_28597_cached_sizze_28811 = 0;
    char *mem_28597 = NULL;
    size_t mem_28635_cached_sizze_28812 = 0;
    char *mem_28635 = NULL;
    size_t mem_28637_cached_sizze_28813 = 0;
    char *mem_28637 = NULL;
    size_t mem_28639_cached_sizze_28814 = 0;
    char *mem_28639 = NULL;
    size_t mem_28641_cached_sizze_28815 = 0;
    char *mem_28641 = NULL;
    size_t mem_28691_cached_sizze_28816 = 0;
    char *mem_28691 = NULL;
    size_t mem_28706_cached_sizze_28817 = 0;
    char *mem_28706 = NULL;
    size_t mem_28718_cached_sizze_28818 = 0;
    char *mem_28718 = NULL;
    size_t mem_28732_cached_sizze_28819 = 0;
    char *mem_28732 = NULL;
    float scalar_out_28762;
    bool bounds_invalid_upwards_28165 = slt64(numswaps_28164, 0);
    bool valid_28166 = !bounds_invalid_upwards_28165;
    bool range_valid_c_28167;
    
    if (!valid_28166) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", numswaps_28164,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cvadynmem.fut:193:29-62\n   #4  cvadynmem.fut:193:19-93\n   #5  cvadynmem.fut:191:1-197:76\n");
        return 1;
    }
    
    float res_28181 = sitofp_i64_f32(numswaps_28164);
    float res_28182 = res_28181 - 1.0F;
    int64_t bytes_28592 = 4 * numswaps_28164;
    
    if (mem_28593_cached_sizze_28809 < (size_t) bytes_28592) {
        mem_28593 = realloc(mem_28593, bytes_28592);
        mem_28593_cached_sizze_28809 = bytes_28592;
    }
    
    int64_t bytes_28594 = 8 * numswaps_28164;
    
    if (mem_28595_cached_sizze_28810 < (size_t) bytes_28594) {
        mem_28595 = realloc(mem_28595, bytes_28594);
        mem_28595_cached_sizze_28810 = bytes_28594;
    }
    if (mem_28597_cached_sizze_28811 < (size_t) bytes_28592) {
        mem_28597 = realloc(mem_28597, bytes_28592);
        mem_28597_cached_sizze_28811 = bytes_28592;
    }
    
    float res_28194;
    float redout_28520 = -INFINITY;
    
    for (int64_t i_28524 = 0; i_28524 < numswaps_28164; i_28524++) {
        int32_t res_28202 = sext_i64_i32(i_28524);
        int32_t x_28203 = lshr32(res_28202, 16);
        int32_t x_28204 = res_28202 ^ x_28203;
        int32_t x_28205 = mul32(73244475, x_28204);
        int32_t x_28206 = lshr32(x_28205, 16);
        int32_t x_28207 = x_28205 ^ x_28206;
        int32_t x_28208 = mul32(73244475, x_28207);
        int32_t x_28209 = lshr32(x_28208, 16);
        int32_t x_28210 = x_28208 ^ x_28209;
        int32_t unsign_arg_28211 = 281253711 ^ x_28210;
        int32_t unsign_arg_28212 = mul32(48271, unsign_arg_28211);
        int32_t unsign_arg_28213 = umod32(unsign_arg_28212, 2147483647);
        float res_28214 = uitofp_i32_f32(unsign_arg_28213);
        float res_28215 = res_28214 / 2.1474836e9F;
        float res_28216 = 2.0F * res_28215;
        float res_28217 = res_28182 * res_28215;
        float res_28218 = 1.0F + res_28217;
        int64_t res_28219 = fptosi_f32_i64(res_28218);
        float res_28225 = -1.0F + res_28216;
        float res_28226 = sitofp_i64_f32(res_28219);
        float res_28227 = res_28216 * res_28226;
        float res_28200 = fmax32(res_28227, redout_28520);
        
        ((float *) mem_28593)[i_28524] = res_28225;
        ((int64_t *) mem_28595)[i_28524] = res_28219;
        ((float *) mem_28597)[i_28524] = res_28216;
        
        float redout_tmp_28763 = res_28200;
        
        redout_28520 = redout_tmp_28763;
    }
    res_28194 = redout_28520;
    
    float res_28232 = sitofp_i64_f32(steps_28163);
    float dt_28233 = res_28194 / res_28232;
    
    if (mem_28635_cached_sizze_28812 < (size_t) bytes_28592) {
        mem_28635 = realloc(mem_28635, bytes_28592);
        mem_28635_cached_sizze_28812 = bytes_28592;
    }
    if (mem_28637_cached_sizze_28813 < (size_t) bytes_28592) {
        mem_28637 = realloc(mem_28637, bytes_28592);
        mem_28637_cached_sizze_28813 = bytes_28592;
    }
    if (mem_28639_cached_sizze_28814 < (size_t) bytes_28594) {
        mem_28639 = realloc(mem_28639, bytes_28594);
        mem_28639_cached_sizze_28814 = bytes_28594;
    }
    if (mem_28641_cached_sizze_28815 < (size_t) bytes_28592) {
        mem_28641 = realloc(mem_28641, bytes_28592);
        mem_28641_cached_sizze_28815 = bytes_28592;
    }
    for (int64_t i_28538 = 0; i_28538 < numswaps_28164; i_28538++) {
        float res_28243 = ((float *) mem_28597)[i_28538];
        int64_t res_28244 = ((int64_t *) mem_28595)[i_28538];
        bool bounds_invalid_upwards_28246 = slt64(res_28244, 1);
        bool valid_28247 = !bounds_invalid_upwards_28246;
        bool range_valid_c_28248;
        
        if (!valid_28247) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 1, "..", 2, "...", res_28244,
                          " is invalid.",
                          "-> #0  cvadynmem.fut:64:29-48\n   #1  cvadynmem.fut:105:25-65\n   #2  cvadynmem.fut:120:16-62\n   #3  cvadynmem.fut:116:17-120:85\n   #4  cvadynmem.fut:197:8-68\n   #5  cvadynmem.fut:191:1-197:76\n");
            return 1;
        }
        
        bool y_28250 = slt64(0, res_28244);
        bool index_certs_28251;
        
        if (!y_28250) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_28244, "].",
                                   "-> #0  cvadynmem.fut:106:47-70\n   #1  cvadynmem.fut:120:16-62\n   #2  cvadynmem.fut:116:17-120:85\n   #3  cvadynmem.fut:197:8-68\n   #4  cvadynmem.fut:191:1-197:76\n");
            return 1;
        }
        
        float binop_y_28252 = sitofp_i64_f32(res_28244);
        float index_primexp_28253 = res_28243 * binop_y_28252;
        float negate_arg_28254 = 1.0e-2F * index_primexp_28253;
        float exp_arg_28255 = 0.0F - negate_arg_28254;
        float res_28256 = fpow32(2.7182817F, exp_arg_28255);
        float x_28257 = 1.0F - res_28256;
        float B_28258 = x_28257 / 1.0e-2F;
        float x_28259 = B_28258 - index_primexp_28253;
        float x_28260 = 4.4999997e-6F * x_28259;
        float A1_28261 = x_28260 / 1.0e-4F;
        float y_28262 = fpow32(B_28258, 2.0F);
        float x_28263 = 1.0000001e-6F * y_28262;
        float A2_28264 = x_28263 / 4.0e-2F;
        float exp_arg_28265 = A1_28261 - A2_28264;
        float res_28266 = fpow32(2.7182817F, exp_arg_28265);
        float negate_arg_28267 = 5.0e-2F * B_28258;
        float exp_arg_28268 = 0.0F - negate_arg_28267;
        float res_28269 = fpow32(2.7182817F, exp_arg_28268);
        float res_28270 = res_28266 * res_28269;
        float res_28271;
        float redout_28528 = 0.0F;
        
        for (int64_t i_28529 = 0; i_28529 < res_28244; i_28529++) {
            int64_t index_primexp_28566 = add64(1, i_28529);
            float res_28276 = sitofp_i64_f32(index_primexp_28566);
            float res_28277 = res_28243 * res_28276;
            float negate_arg_28278 = 1.0e-2F * res_28277;
            float exp_arg_28279 = 0.0F - negate_arg_28278;
            float res_28280 = fpow32(2.7182817F, exp_arg_28279);
            float x_28281 = 1.0F - res_28280;
            float B_28282 = x_28281 / 1.0e-2F;
            float x_28283 = B_28282 - res_28277;
            float x_28284 = 4.4999997e-6F * x_28283;
            float A1_28285 = x_28284 / 1.0e-4F;
            float y_28286 = fpow32(B_28282, 2.0F);
            float x_28287 = 1.0000001e-6F * y_28286;
            float A2_28288 = x_28287 / 4.0e-2F;
            float exp_arg_28289 = A1_28285 - A2_28288;
            float res_28290 = fpow32(2.7182817F, exp_arg_28289);
            float negate_arg_28291 = 5.0e-2F * B_28282;
            float exp_arg_28292 = 0.0F - negate_arg_28291;
            float res_28293 = fpow32(2.7182817F, exp_arg_28292);
            float res_28294 = res_28290 * res_28293;
            float res_28274 = res_28294 + redout_28528;
            float redout_tmp_28771 = res_28274;
            
            redout_28528 = redout_tmp_28771;
        }
        res_28271 = redout_28528;
        
        float x_28295 = 1.0F - res_28270;
        float y_28296 = res_28243 * res_28271;
        float res_28297 = x_28295 / y_28296;
        
        ((float *) mem_28635)[i_28538] = res_28297;
        memmove(mem_28637 + i_28538 * 4, mem_28593 + i_28538 * 4,
                (int32_t) sizeof(float));
        memmove(mem_28639 + i_28538 * 8, mem_28595 + i_28538 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_28641 + i_28538 * 4, mem_28597 + i_28538 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_28298 = res_28232 / res_28194;
    bool bounds_invalid_upwards_28299 = slt64(steps_28163, 1);
    bool valid_28300 = !bounds_invalid_upwards_28299;
    bool range_valid_c_28301;
    
    if (!valid_28300) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_28163,
                               " is invalid.",
                               "-> #0  cvadynmem.fut:70:56-67\n   #1  cvadynmem.fut:122:17-44\n   #2  cvadynmem.fut:197:8-68\n   #3  cvadynmem.fut:191:1-197:76\n");
        return 1;
    }
    
    int64_t bytes_28690 = 4 * steps_28163;
    
    if (mem_28691_cached_sizze_28816 < (size_t) bytes_28690) {
        mem_28691 = realloc(mem_28691, bytes_28690);
        mem_28691_cached_sizze_28816 = bytes_28690;
    }
    for (int64_t i_28545 = 0; i_28545 < steps_28163; i_28545++) {
        int64_t index_primexp_28573 = add64(1, i_28545);
        float res_28305 = sitofp_i64_f32(index_primexp_28573);
        float res_28306 = res_28305 / sims_per_year_28298;
        
        ((float *) mem_28691)[i_28545] = res_28306;
    }
    
    bool bounds_invalid_upwards_28307 = slt64(paths_28162, 0);
    bool valid_28308 = !bounds_invalid_upwards_28307;
    bool range_valid_c_28309;
    
    if (!valid_28308) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_28162,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cvadynmem.fut:125:19-49\n   #4  cvadynmem.fut:197:8-68\n   #5  cvadynmem.fut:191:1-197:76\n");
        return 1;
    }
    
    int64_t upper_bound_28312 = sub64(steps_28163, 1);
    float res_28313;
    
    res_28313 = futrts_sqrt32(dt_28233);
    
    int64_t binop_x_28705 = paths_28162 * steps_28163;
    int64_t bytes_28704 = 4 * binop_x_28705;
    
    if (mem_28706_cached_sizze_28817 < (size_t) bytes_28704) {
        mem_28706 = realloc(mem_28706, bytes_28704);
        mem_28706_cached_sizze_28817 = bytes_28704;
    }
    if (mem_28718_cached_sizze_28818 < (size_t) bytes_28690) {
        mem_28718 = realloc(mem_28718, bytes_28690);
        mem_28718_cached_sizze_28818 = bytes_28690;
    }
    if (mem_28732_cached_sizze_28819 < (size_t) bytes_28690) {
        mem_28732 = realloc(mem_28732, bytes_28690);
        mem_28732_cached_sizze_28819 = bytes_28690;
    }
    for (int64_t i_28773 = 0; i_28773 < steps_28163; i_28773++) {
        ((float *) mem_28732)[i_28773] = 5.0e-2F;
    }
    for (int64_t i_28553 = 0; i_28553 < paths_28162; i_28553++) {
        int32_t res_28316 = sext_i64_i32(i_28553);
        int32_t x_28317 = lshr32(res_28316, 16);
        int32_t x_28318 = res_28316 ^ x_28317;
        int32_t x_28319 = mul32(73244475, x_28318);
        int32_t x_28320 = lshr32(x_28319, 16);
        int32_t x_28321 = x_28319 ^ x_28320;
        int32_t x_28322 = mul32(73244475, x_28321);
        int32_t x_28323 = lshr32(x_28322, 16);
        int32_t x_28324 = x_28322 ^ x_28323;
        int32_t unsign_arg_28325 = 777822902 ^ x_28324;
        int32_t unsign_arg_28326 = mul32(48271, unsign_arg_28325);
        int32_t unsign_arg_28327 = umod32(unsign_arg_28326, 2147483647);
        
        for (int64_t i_28549 = 0; i_28549 < steps_28163; i_28549++) {
            int32_t res_28330 = sext_i64_i32(i_28549);
            int32_t x_28331 = lshr32(res_28330, 16);
            int32_t x_28332 = res_28330 ^ x_28331;
            int32_t x_28333 = mul32(73244475, x_28332);
            int32_t x_28334 = lshr32(x_28333, 16);
            int32_t x_28335 = x_28333 ^ x_28334;
            int32_t x_28336 = mul32(73244475, x_28335);
            int32_t x_28337 = lshr32(x_28336, 16);
            int32_t x_28338 = x_28336 ^ x_28337;
            int32_t unsign_arg_28339 = unsign_arg_28327 ^ x_28338;
            int32_t unsign_arg_28340 = mul32(48271, unsign_arg_28339);
            int32_t unsign_arg_28341 = umod32(unsign_arg_28340, 2147483647);
            int32_t unsign_arg_28342 = mul32(48271, unsign_arg_28341);
            int32_t unsign_arg_28343 = umod32(unsign_arg_28342, 2147483647);
            float res_28344 = uitofp_i32_f32(unsign_arg_28341);
            float res_28345 = res_28344 / 2.1474836e9F;
            float res_28346 = uitofp_i32_f32(unsign_arg_28343);
            float res_28347 = res_28346 / 2.1474836e9F;
            float res_28348;
            
            res_28348 = futrts_log32(res_28345);
            
            float res_28349 = -2.0F * res_28348;
            float res_28350;
            
            res_28350 = futrts_sqrt32(res_28349);
            
            float res_28351 = 6.2831855F * res_28347;
            float res_28352;
            
            res_28352 = futrts_cos32(res_28351);
            
            float res_28353 = res_28350 * res_28352;
            
            ((float *) mem_28718)[i_28549] = res_28353;
        }
        memmove(mem_28706 + i_28553 * steps_28163 * 4, mem_28732 + 0,
                steps_28163 * (int32_t) sizeof(float));
        for (int64_t i_28356 = 0; i_28356 < upper_bound_28312; i_28356++) {
            bool y_28358 = slt64(i_28356, steps_28163);
            bool index_certs_28359;
            
            if (!y_28358) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_28356,
                              "] out of bounds for array of shape [",
                              steps_28163, "].",
                              "-> #0  cvadynmem.fut:81:97-104\n   #1  cvadynmem.fut:133:32-62\n   #2  cvadynmem.fut:133:22-69\n   #3  cvadynmem.fut:197:8-68\n   #4  cvadynmem.fut:191:1-197:76\n");
                return 1;
            }
            
            float shortstep_arg_28360 = ((float *) mem_28718)[i_28356];
            float shortstep_arg_28361 = ((float *) mem_28706)[i_28553 *
                                                              steps_28163 +
                                                              i_28356];
            float y_28362 = 5.0e-2F - shortstep_arg_28361;
            float x_28363 = 1.0e-2F * y_28362;
            float x_28364 = dt_28233 * x_28363;
            float x_28365 = res_28313 * shortstep_arg_28360;
            float y_28366 = 1.0e-3F * x_28365;
            float delta_r_28367 = x_28364 + y_28366;
            float res_28368 = shortstep_arg_28361 + delta_r_28367;
            int64_t i_28369 = add64(1, i_28356);
            bool x_28370 = sle64(0, i_28369);
            bool y_28371 = slt64(i_28369, steps_28163);
            bool bounds_check_28372 = x_28370 && y_28371;
            bool index_certs_28373;
            
            if (!bounds_check_28372) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_28369,
                              "] out of bounds for array of shape [",
                              steps_28163, "].",
                              "-> #0  cvadynmem.fut:81:58-105\n   #1  cvadynmem.fut:133:32-62\n   #2  cvadynmem.fut:133:22-69\n   #3  cvadynmem.fut:197:8-68\n   #4  cvadynmem.fut:191:1-197:76\n");
                return 1;
            }
            ((float *) mem_28706)[i_28553 * steps_28163 + i_28369] = res_28368;
        }
    }
    
    float res_28375 = sitofp_i64_f32(paths_28162);
    bool loop_nonempty_28580 = slt64(0, paths_28162);
    float res_28382;
    float redout_28561 = 0.0F;
    
    for (int64_t i_28562 = 0; i_28562 < steps_28163; i_28562++) {
        int64_t index_primexp_28584 = add64(1, i_28562);
        float res_28389 = sitofp_i64_f32(index_primexp_28584);
        float res_28390 = res_28389 / sims_per_year_28298;
        float x_28396;
        
        if (loop_nonempty_28580) {
            float x_28581 = ((float *) mem_28691)[i_28562];
            
            x_28396 = x_28581;
        } else {
            x_28396 = 0.0F;
        }
        
        float res_28391;
        float redout_28559 = 0.0F;
        
        for (int64_t i_28560 = 0; i_28560 < paths_28162; i_28560++) {
            float x_28395 = ((float *) mem_28706)[i_28560 * steps_28163 +
                                                  i_28562];
            float res_28397;
            float redout_28557 = 0.0F;
            
            for (int64_t i_28558 = 0; i_28558 < numswaps_28164; i_28558++) {
                float x_28401 = ((float *) mem_28635)[i_28558];
                float x_28402 = ((float *) mem_28637)[i_28558];
                int64_t x_28403 = ((int64_t *) mem_28639)[i_28558];
                float x_28404 = ((float *) mem_28641)[i_28558];
                int64_t i64_arg_28405 = sub64(x_28403, 1);
                float res_28406 = sitofp_i64_f32(i64_arg_28405);
                float y_28407 = x_28404 * res_28406;
                bool cond_28408 = y_28407 < x_28396;
                float ceil_arg_28409 = x_28396 / x_28404;
                float res_28410;
                
                res_28410 = futrts_ceil32(ceil_arg_28409);
                
                int64_t res_28411 = fptosi_f32_i64(res_28410);
                int64_t remaining_28412 = sub64(x_28403, res_28411);
                float res_28413;
                
                if (cond_28408) {
                    res_28413 = 0.0F;
                } else {
                    float nextpayment_28414 = x_28404 * res_28410;
                    bool bounds_invalid_upwards_28415 = slt64(remaining_28412,
                                                              1);
                    bool valid_28416 = !bounds_invalid_upwards_28415;
                    bool range_valid_c_28417;
                    
                    if (!valid_28416) {
                        ctx->error =
                            msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                                      "Range ", 1, "..", 2, "...",
                                      remaining_28412, " is invalid.",
                                      "-> #0  cvadynmem.fut:40:30-45\n   #1  cvadynmem.fut:48:27-71\n   #2  cvadynmem.fut:140:51-76\n   #3  cvadynmem.fut:138:35-140:84\n   #4  /prelude/soacs.fut:56:19-23\n   #5  /prelude/soacs.fut:56:3-37\n   #6  cvadynmem.fut:137:25-144:33\n   #7  cvadynmem.fut:136:21-144:45\n   #8  cvadynmem.fut:197:8-68\n   #9  cvadynmem.fut:191:1-197:76\n");
                        return 1;
                    }
                    
                    float y_28419 = nextpayment_28414 - x_28396;
                    float negate_arg_28420 = 1.0e-2F * y_28419;
                    float exp_arg_28421 = 0.0F - negate_arg_28420;
                    float res_28422 = fpow32(2.7182817F, exp_arg_28421);
                    float x_28423 = 1.0F - res_28422;
                    float B_28424 = x_28423 / 1.0e-2F;
                    float x_28425 = B_28424 - nextpayment_28414;
                    float x_28426 = x_28396 + x_28425;
                    float x_28427 = 4.4999997e-6F * x_28426;
                    float A1_28428 = x_28427 / 1.0e-4F;
                    float y_28429 = fpow32(B_28424, 2.0F);
                    float x_28430 = 1.0000001e-6F * y_28429;
                    float A2_28431 = x_28430 / 4.0e-2F;
                    float exp_arg_28432 = A1_28428 - A2_28431;
                    float res_28433 = fpow32(2.7182817F, exp_arg_28432);
                    float negate_arg_28434 = x_28395 * B_28424;
                    float exp_arg_28435 = 0.0F - negate_arg_28434;
                    float res_28436 = fpow32(2.7182817F, exp_arg_28435);
                    float res_28437 = res_28433 * res_28436;
                    bool y_28438 = slt64(0, remaining_28412);
                    bool index_certs_28439;
                    
                    if (!y_28438) {
                        ctx->error =
                            msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                      "Index [", 0,
                                      "] out of bounds for array of shape [",
                                      remaining_28412, "].",
                                      "-> #0  cvadynmem.fut:50:38-63\n   #1  cvadynmem.fut:140:51-76\n   #2  cvadynmem.fut:138:35-140:84\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  cvadynmem.fut:137:25-144:33\n   #6  cvadynmem.fut:136:21-144:45\n   #7  cvadynmem.fut:197:8-68\n   #8  cvadynmem.fut:191:1-197:76\n");
                        return 1;
                    }
                    
                    float binop_y_28440 = sitofp_i64_f32(remaining_28412);
                    float binop_y_28441 = x_28404 * binop_y_28440;
                    float index_primexp_28442 = nextpayment_28414 +
                          binop_y_28441;
                    float y_28443 = index_primexp_28442 - x_28396;
                    float negate_arg_28444 = 1.0e-2F * y_28443;
                    float exp_arg_28445 = 0.0F - negate_arg_28444;
                    float res_28446 = fpow32(2.7182817F, exp_arg_28445);
                    float x_28447 = 1.0F - res_28446;
                    float B_28448 = x_28447 / 1.0e-2F;
                    float x_28449 = B_28448 - index_primexp_28442;
                    float x_28450 = x_28396 + x_28449;
                    float x_28451 = 4.4999997e-6F * x_28450;
                    float A1_28452 = x_28451 / 1.0e-4F;
                    float y_28453 = fpow32(B_28448, 2.0F);
                    float x_28454 = 1.0000001e-6F * y_28453;
                    float A2_28455 = x_28454 / 4.0e-2F;
                    float exp_arg_28456 = A1_28452 - A2_28455;
                    float res_28457 = fpow32(2.7182817F, exp_arg_28456);
                    float negate_arg_28458 = x_28395 * B_28448;
                    float exp_arg_28459 = 0.0F - negate_arg_28458;
                    float res_28460 = fpow32(2.7182817F, exp_arg_28459);
                    float res_28461 = res_28457 * res_28460;
                    float res_28462;
                    float redout_28555 = 0.0F;
                    
                    for (int64_t i_28556 = 0; i_28556 < remaining_28412;
                         i_28556++) {
                        int64_t index_primexp_28579 = add64(1, i_28556);
                        float res_28467 = sitofp_i64_f32(index_primexp_28579);
                        float res_28468 = x_28404 * res_28467;
                        float res_28469 = nextpayment_28414 + res_28468;
                        float y_28470 = res_28469 - x_28396;
                        float negate_arg_28471 = 1.0e-2F * y_28470;
                        float exp_arg_28472 = 0.0F - negate_arg_28471;
                        float res_28473 = fpow32(2.7182817F, exp_arg_28472);
                        float x_28474 = 1.0F - res_28473;
                        float B_28475 = x_28474 / 1.0e-2F;
                        float x_28476 = B_28475 - res_28469;
                        float x_28477 = x_28396 + x_28476;
                        float x_28478 = 4.4999997e-6F * x_28477;
                        float A1_28479 = x_28478 / 1.0e-4F;
                        float y_28480 = fpow32(B_28475, 2.0F);
                        float x_28481 = 1.0000001e-6F * y_28480;
                        float A2_28482 = x_28481 / 4.0e-2F;
                        float exp_arg_28483 = A1_28479 - A2_28482;
                        float res_28484 = fpow32(2.7182817F, exp_arg_28483);
                        float negate_arg_28485 = x_28395 * B_28475;
                        float exp_arg_28486 = 0.0F - negate_arg_28485;
                        float res_28487 = fpow32(2.7182817F, exp_arg_28486);
                        float res_28488 = res_28484 * res_28487;
                        float res_28465 = res_28488 + redout_28555;
                        float redout_tmp_28780 = res_28465;
                        
                        redout_28555 = redout_tmp_28780;
                    }
                    res_28462 = redout_28555;
                    
                    float x_28489 = res_28437 - res_28461;
                    float x_28490 = x_28401 * x_28404;
                    float y_28491 = res_28462 * x_28490;
                    float y_28492 = x_28489 - y_28491;
                    float res_28493 = x_28402 * y_28492;
                    float res_28494 = x_28402 * res_28493;
                    
                    res_28413 = res_28494;
                }
                
                float res_28400 = res_28413 + redout_28557;
                float redout_tmp_28779 = res_28400;
                
                redout_28557 = redout_tmp_28779;
            }
            res_28397 = redout_28557;
            
            float res_28495 = fmax32(0.0F, res_28397);
            float res_28394 = res_28495 + redout_28559;
            float redout_tmp_28778 = res_28394;
            
            redout_28559 = redout_tmp_28778;
        }
        res_28391 = redout_28559;
        
        float res_28496 = res_28391 / res_28375;
        float negate_arg_28497 = 1.0e-2F * res_28390;
        float exp_arg_28498 = 0.0F - negate_arg_28497;
        float res_28499 = fpow32(2.7182817F, exp_arg_28498);
        float x_28500 = 1.0F - res_28499;
        float B_28501 = x_28500 / 1.0e-2F;
        float x_28502 = B_28501 - res_28390;
        float x_28503 = 4.4999997e-6F * x_28502;
        float A1_28504 = x_28503 / 1.0e-4F;
        float y_28505 = fpow32(B_28501, 2.0F);
        float x_28506 = 1.0000001e-6F * y_28505;
        float A2_28507 = x_28506 / 4.0e-2F;
        float exp_arg_28508 = A1_28504 - A2_28507;
        float res_28509 = fpow32(2.7182817F, exp_arg_28508);
        float negate_arg_28510 = 5.0e-2F * B_28501;
        float exp_arg_28511 = 0.0F - negate_arg_28510;
        float res_28512 = fpow32(2.7182817F, exp_arg_28511);
        float res_28513 = res_28509 * res_28512;
        float res_28514 = res_28496 * res_28513;
        float res_28385 = res_28514 + redout_28561;
        float redout_tmp_28777 = res_28385;
        
        redout_28561 = redout_tmp_28777;
    }
    res_28382 = redout_28561;
    
    float CVA_28516 = 6.0e-3F * res_28382;
    
    scalar_out_28762 = CVA_28516;
    *out_scalar_out_28808 = scalar_out_28762;
    
  cleanup:
    { }
    free(mem_28593);
    free(mem_28595);
    free(mem_28597);
    free(mem_28635);
    free(mem_28637);
    free(mem_28639);
    free(mem_28641);
    free(mem_28691);
    free(mem_28706);
    free(mem_28718);
    free(mem_28732);
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
    struct memblock swap_term_mem_28592;
    
    swap_term_mem_28592.references = NULL;
    
    struct memblock payments_mem_28593;
    
    payments_mem_28593.references = NULL;
    
    struct memblock notional_mem_28594;
    
    notional_mem_28594.references = NULL;
    
    int64_t n_27423;
    int64_t n_27424;
    int64_t n_27425;
    int64_t paths_27426;
    int64_t steps_27427;
    float a_27431;
    float b_27432;
    float sigma_27433;
    float r0_27434;
    float scalar_out_28762;
    struct memblock out_mem_28763;
    
    out_mem_28763.references = NULL;
    
    int64_t out_arrsizze_28764;
    
    lock_lock(&ctx->lock);
    paths_27426 = in0;
    steps_27427 = in1;
    swap_term_mem_28592 = in2->mem;
    n_27423 = in2->shape[0];
    payments_mem_28593 = in3->mem;
    n_27424 = in3->shape[0];
    notional_mem_28594 = in4->mem;
    n_27425 = in4->shape[0];
    a_27431 = in5;
    b_27432 = in6;
    sigma_27433 = in7;
    r0_27434 = in8;
    
    int ret = futrts_main(ctx, &scalar_out_28762, &out_mem_28763,
                          &out_arrsizze_28764, swap_term_mem_28592,
                          payments_mem_28593, notional_mem_28594, n_27423,
                          n_27424, n_27425, paths_27426, steps_27427, a_27431,
                          b_27432, sigma_27433, r0_27434);
    
    if (ret == 0) {
        *out0 = scalar_out_28762;
        assert((*out1 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out1)->mem = out_mem_28763;
        (*out1)->shape[0] = out_arrsizze_28764;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test(struct futhark_context *ctx, float *out0, const
                       int64_t in0, const int64_t in1)
{
    int64_t paths_27863;
    int64_t steps_27864;
    float scalar_out_28762;
    
    lock_lock(&ctx->lock);
    paths_27863 = in0;
    steps_27864 = in1;
    
    int ret = futrts_test(ctx, &scalar_out_28762, paths_27863, steps_27864);
    
    if (ret == 0) {
        *out0 = scalar_out_28762;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test2(struct futhark_context *ctx, float *out0, const
                        int64_t in0, const int64_t in1, const int64_t in2)
{
    int64_t paths_28162;
    int64_t steps_28163;
    int64_t numswaps_28164;
    float scalar_out_28762;
    
    lock_lock(&ctx->lock);
    paths_28162 = in0;
    steps_28163 = in1;
    numswaps_28164 = in2;
    
    int ret = futrts_test2(ctx, &scalar_out_28762, paths_28162, steps_28163,
                           numswaps_28164);
    
    if (ret == 0) {
        *out0 = scalar_out_28762;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
