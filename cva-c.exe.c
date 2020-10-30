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
    
    int64_t read_value_26530;
    
    if (read_scalar(&i64_info, &read_value_26530) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_26531;
    
    if (read_scalar(&i64_info, &read_value_26531) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_26532;
    int64_t read_shape_26533[1];
    float *read_arr_26534 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26534, read_shape_26533, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
                      "[]", f32_info.type_name, strerror(errno));
    
    struct futhark_i64_1d *read_value_26535;
    int64_t read_shape_26536[1];
    int64_t *read_arr_26537 = NULL;
    
    errno = 0;
    if (read_array(&i64_info, (void **) &read_arr_26537, read_shape_26536, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
                      "[]", i64_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_26538;
    int64_t read_shape_26539[1];
    float *read_arr_26540 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26540, read_shape_26539, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4,
                      "[]", f32_info.type_name, strerror(errno));
    
    float read_value_26541;
    
    if (read_scalar(&f32_info, &read_value_26541) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      5, f32_info.type_name, strerror(errno));
    
    float read_value_26542;
    
    if (read_scalar(&f32_info, &read_value_26542) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      6, f32_info.type_name, strerror(errno));
    
    float read_value_26543;
    
    if (read_scalar(&f32_info, &read_value_26543) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      7, f32_info.type_name, strerror(errno));
    
    float read_value_26544;
    
    if (read_scalar(&f32_info, &read_value_26544) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      8, f32_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"main\"");
    
    float result_26545;
    struct futhark_f32_1d *result_26546;
    
    if (perform_warmup) {
        int r;
        
        ;
        ;
        assert((read_value_26532 = futhark_new_f32_1d(ctx, read_arr_26534,
                                                      read_shape_26533[0])) !=
            0);
        assert((read_value_26535 = futhark_new_i64_1d(ctx, read_arr_26537,
                                                      read_shape_26536[0])) !=
            0);
        assert((read_value_26538 = futhark_new_f32_1d(ctx, read_arr_26540,
                                                      read_shape_26539[0])) !=
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
        r = futhark_entry_main(ctx, &result_26545, &result_26546,
                               read_value_26530, read_value_26531,
                               read_value_26532, read_value_26535,
                               read_value_26538, read_value_26541,
                               read_value_26542, read_value_26543,
                               read_value_26544);
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
        assert(futhark_free_f32_1d(ctx, read_value_26532) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_26535) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_26538) == 0);
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_f32_1d(ctx, result_26546) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        ;
        assert((read_value_26532 = futhark_new_f32_1d(ctx, read_arr_26534,
                                                      read_shape_26533[0])) !=
            0);
        assert((read_value_26535 = futhark_new_i64_1d(ctx, read_arr_26537,
                                                      read_shape_26536[0])) !=
            0);
        assert((read_value_26538 = futhark_new_f32_1d(ctx, read_arr_26540,
                                                      read_shape_26539[0])) !=
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
        r = futhark_entry_main(ctx, &result_26545, &result_26546,
                               read_value_26530, read_value_26531,
                               read_value_26532, read_value_26535,
                               read_value_26538, read_value_26541,
                               read_value_26542, read_value_26543,
                               read_value_26544);
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
        assert(futhark_free_f32_1d(ctx, read_value_26532) == 0);
        assert(futhark_free_i64_1d(ctx, read_value_26535) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_26538) == 0);
        ;
        ;
        ;
        ;
        if (run < num_runs - 1) {
            ;
            assert(futhark_free_f32_1d(ctx, result_26546) == 0);
        }
    }
    ;
    ;
    free(read_arr_26534);
    free(read_arr_26537);
    free(read_arr_26540);
    ;
    ;
    ;
    ;
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &f32_info, &result_26545);
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_26546)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_26546, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_26546), 1);
        free(arr);
    }
    printf("\n");
    ;
    assert(futhark_free_f32_1d(ctx, result_26546) == 0);
}
static void futrts_cli_entry_test(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    int64_t read_value_26547;
    
    if (read_scalar(&i64_info, &read_value_26547) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i64_info.type_name, strerror(errno));
    
    int64_t read_value_26548;
    
    if (read_scalar(&i64_info, &read_value_26548) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      1, i64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"test\"");
    
    float result_26549;
    
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
        r = futhark_entry_test(ctx, &result_26549, read_value_26547,
                               read_value_26548);
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
        r = futhark_entry_test(ctx, &result_26549, read_value_26547,
                               read_value_26548);
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
    write_scalar(stdout, binary_output, &f32_info, &result_26549);
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
                                                                         futrts_cli_entry_test}};
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
static float testzistatic_array_realtype_26527[9] = {1.0F, -0.5F, 1.0F, 1.0F,
                                                     1.0F, 1.0F, 1.0F, 1.0F,
                                                     1.0F};
static int64_t testzistatic_array_realtype_26528[9] = {10, 20, 5, 5, 50, 20, 30,
                                                       15, 18};
static float testzistatic_array_realtype_26529[9] = {1.0F, 0.5F, 0.25F, 0.1F,
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
    struct memblock testzistatic_array_26448;
    struct memblock testzistatic_array_26449;
    struct memblock testzistatic_array_26450;
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
    ctx->testzistatic_array_26448 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_26527,
                                                       0};
    ctx->testzistatic_array_26449 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_26528,
                                                       0};
    ctx->testzistatic_array_26450 = (struct memblock) {NULL,
                                                       (char *) testzistatic_array_realtype_26529,
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
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_26483,
                       struct memblock *out_mem_p_26484,
                       int64_t *out_out_arrsizze_26485,
                       struct memblock swap_term_mem_26157,
                       struct memblock payments_mem_26158,
                       struct memblock notional_mem_26159, int64_t n_24721,
                       int64_t n_24722, int64_t n_24723, int64_t paths_24724,
                       int64_t steps_24725, float a_24729, float b_24730,
                       float sigma_24731, float r0_24732);
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_26505,
                       int64_t paths_25438, int64_t steps_25439);
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
static int futrts_main(struct futhark_context *ctx, float *out_scalar_out_26483,
                       struct memblock *out_mem_p_26484,
                       int64_t *out_out_arrsizze_26485,
                       struct memblock swap_term_mem_26157,
                       struct memblock payments_mem_26158,
                       struct memblock notional_mem_26159, int64_t n_24721,
                       int64_t n_24722, int64_t n_24723, int64_t paths_24724,
                       int64_t steps_24725, float a_24729, float b_24730,
                       float sigma_24731, float r0_24732)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_26161_cached_sizze_26486 = 0;
    char *mem_26161 = NULL;
    size_t mem_26163_cached_sizze_26487 = 0;
    char *mem_26163 = NULL;
    size_t mem_26165_cached_sizze_26488 = 0;
    char *mem_26165 = NULL;
    size_t mem_26167_cached_sizze_26489 = 0;
    char *mem_26167 = NULL;
    size_t mem_26197_cached_sizze_26490 = 0;
    char *mem_26197 = NULL;
    size_t mem_26232_cached_sizze_26491 = 0;
    char *mem_26232 = NULL;
    size_t mem_26244_cached_sizze_26492 = 0;
    char *mem_26244 = NULL;
    size_t mem_26258_cached_sizze_26493 = 0;
    char *mem_26258 = NULL;
    size_t mem_26286_cached_sizze_26494 = 0;
    char *mem_26286 = NULL;
    size_t mem_26296_cached_sizze_26495 = 0;
    char *mem_26296 = NULL;
    size_t mem_26308_cached_sizze_26496 = 0;
    char *mem_26308 = NULL;
    size_t mem_26318_cached_sizze_26497 = 0;
    char *mem_26318 = NULL;
    size_t mem_26320_cached_sizze_26498 = 0;
    char *mem_26320 = NULL;
    size_t mem_26346_cached_sizze_26499 = 0;
    char *mem_26346 = NULL;
    size_t mem_26360_cached_sizze_26500 = 0;
    char *mem_26360 = NULL;
    size_t mem_26374_cached_sizze_26501 = 0;
    char *mem_26374 = NULL;
    size_t mem_26376_cached_sizze_26502 = 0;
    char *mem_26376 = NULL;
    size_t mem_26402_cached_sizze_26503 = 0;
    char *mem_26402 = NULL;
    size_t mem_26416_cached_sizze_26504 = 0;
    char *mem_26416 = NULL;
    float scalar_out_26447;
    struct memblock out_mem_26448;
    
    out_mem_26448.references = NULL;
    
    int64_t out_arrsizze_26449;
    bool dim_match_24733 = n_24721 == n_24722;
    bool empty_or_match_cert_24734;
    
    if (!dim_match_24733) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cva.fut:100:1-165:21\n");
        if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_24735 = n_24721 == n_24723;
    bool empty_or_match_cert_24736;
    
    if (!dim_match_24735) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  cva.fut:100:1-165:21\n");
        if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
            return 1;
        return 1;
    }
    
    float res_24738;
    float redout_25876 = -INFINITY;
    
    for (int64_t i_25877 = 0; i_25877 < n_24721; i_25877++) {
        float x_24742 = ((float *) swap_term_mem_26157.mem)[i_25877];
        int64_t x_24743 = ((int64_t *) payments_mem_26158.mem)[i_25877];
        float res_24744 = sitofp_i64_f32(x_24743);
        float res_24745 = x_24742 * res_24744;
        float res_24741 = fmax32(res_24745, redout_25876);
        float redout_tmp_26450 = res_24741;
        
        redout_25876 = redout_tmp_26450;
    }
    res_24738 = redout_25876;
    
    float res_24746 = sitofp_i64_f32(steps_24725);
    float dt_24747 = res_24738 / res_24746;
    float x_24749 = fpow32(a_24729, 2.0F);
    float x_24750 = b_24730 * x_24749;
    float x_24751 = fpow32(sigma_24731, 2.0F);
    float y_24752 = x_24751 / 2.0F;
    float y_24753 = x_24750 - y_24752;
    float y_24754 = 4.0F * a_24729;
    int64_t bytes_26160 = 4 * n_24721;
    
    if (mem_26161_cached_sizze_26486 < (size_t) bytes_26160) {
        mem_26161 = realloc(mem_26161, bytes_26160);
        mem_26161_cached_sizze_26486 = bytes_26160;
    }
    if (mem_26163_cached_sizze_26487 < (size_t) bytes_26160) {
        mem_26163 = realloc(mem_26163, bytes_26160);
        mem_26163_cached_sizze_26487 = bytes_26160;
    }
    
    int64_t bytes_26164 = 8 * n_24721;
    
    if (mem_26165_cached_sizze_26488 < (size_t) bytes_26164) {
        mem_26165 = realloc(mem_26165, bytes_26164);
        mem_26165_cached_sizze_26488 = bytes_26164;
    }
    if (mem_26167_cached_sizze_26489 < (size_t) bytes_26160) {
        mem_26167 = realloc(mem_26167, bytes_26160);
        mem_26167_cached_sizze_26489 = bytes_26160;
    }
    for (int64_t i_25892 = 0; i_25892 < n_24721; i_25892++) {
        float res_24764 = ((float *) swap_term_mem_26157.mem)[i_25892];
        int64_t res_24765 = ((int64_t *) payments_mem_26158.mem)[i_25892];
        int64_t range_end_24767 = sub64(res_24765, 1);
        bool bounds_invalid_upwards_24768 = slt64(range_end_24767, 0);
        bool valid_24769 = !bounds_invalid_upwards_24768;
        bool range_valid_c_24770;
        
        if (!valid_24769) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "...", range_end_24767,
                          " is invalid.",
                          "-> #0  cva.fut:54:29-52\n   #1  cva.fut:95:25-65\n   #2  cva.fut:109:16-62\n   #3  cva.fut:105:17-109:85\n   #4  cva.fut:100:1-165:21\n");
            if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                return 1;
            return 1;
        }
        
        int64_t bytes_26196 = 4 * res_24765;
        
        if (mem_26197_cached_sizze_26490 < (size_t) bytes_26196) {
            mem_26197 = realloc(mem_26197, bytes_26196);
            mem_26197_cached_sizze_26490 = bytes_26196;
        }
        for (int64_t i_25880 = 0; i_25880 < res_24765; i_25880++) {
            float res_24774 = sitofp_i64_f32(i_25880);
            float res_24775 = res_24764 * res_24774;
            
            ((float *) mem_26197)[i_25880] = res_24775;
        }
        
        bool y_24776 = slt64(0, res_24765);
        bool index_certs_24777;
        
        if (!y_24776) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_24765, "].",
                                   "-> #0  cva.fut:96:47-70\n   #1  cva.fut:109:16-62\n   #2  cva.fut:105:17-109:85\n   #3  cva.fut:100:1-165:21\n");
            if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                return 1;
            return 1;
        }
        
        float binop_y_24778 = sitofp_i64_f32(range_end_24767);
        float index_primexp_24779 = res_24764 * binop_y_24778;
        float negate_arg_24780 = a_24729 * index_primexp_24779;
        float exp_arg_24781 = 0.0F - negate_arg_24780;
        float res_24782 = fpow32(2.7182817F, exp_arg_24781);
        float x_24783 = 1.0F - res_24782;
        float B_24784 = x_24783 / a_24729;
        float x_24785 = B_24784 - index_primexp_24779;
        float x_24786 = y_24753 * x_24785;
        float A1_24787 = x_24786 / x_24749;
        float y_24788 = fpow32(B_24784, 2.0F);
        float x_24789 = x_24751 * y_24788;
        float A2_24790 = x_24789 / y_24754;
        float exp_arg_24791 = A1_24787 - A2_24790;
        float res_24792 = fpow32(2.7182817F, exp_arg_24791);
        float negate_arg_24793 = r0_24732 * B_24784;
        float exp_arg_24794 = 0.0F - negate_arg_24793;
        float res_24795 = fpow32(2.7182817F, exp_arg_24794);
        float res_24796 = res_24792 * res_24795;
        bool empty_slice_24797 = range_end_24767 == 0;
        bool zzero_leq_i_p_m_t_s_24798 = sle64(0, range_end_24767);
        bool i_p_m_t_s_leq_w_24799 = slt64(range_end_24767, res_24765);
        bool i_lte_j_24800 = sle64(1, res_24765);
        bool y_24801 = zzero_leq_i_p_m_t_s_24798 && i_p_m_t_s_leq_w_24799;
        bool y_24802 = i_lte_j_24800 && y_24801;
        bool ok_or_empty_24803 = empty_slice_24797 || y_24802;
        bool index_certs_24804;
        
        if (!ok_or_empty_24803) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 1,
                                   ":] out of bounds for array of shape [",
                                   res_24765, "].",
                                   "-> #0  cva.fut:97:74-90\n   #1  cva.fut:109:16-62\n   #2  cva.fut:105:17-109:85\n   #3  cva.fut:100:1-165:21\n");
            if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                return 1;
            return 1;
        }
        
        float res_24806;
        float redout_25882 = 0.0F;
        
        for (int64_t i_25883 = 0; i_25883 < range_end_24767; i_25883++) {
            int64_t slice_26012 = 1 + i_25883;
            float x_24810 = ((float *) mem_26197)[slice_26012];
            float negate_arg_24811 = a_24729 * x_24810;
            float exp_arg_24812 = 0.0F - negate_arg_24811;
            float res_24813 = fpow32(2.7182817F, exp_arg_24812);
            float x_24814 = 1.0F - res_24813;
            float B_24815 = x_24814 / a_24729;
            float x_24816 = B_24815 - x_24810;
            float x_24817 = y_24753 * x_24816;
            float A1_24818 = x_24817 / x_24749;
            float y_24819 = fpow32(B_24815, 2.0F);
            float x_24820 = x_24751 * y_24819;
            float A2_24821 = x_24820 / y_24754;
            float exp_arg_24822 = A1_24818 - A2_24821;
            float res_24823 = fpow32(2.7182817F, exp_arg_24822);
            float negate_arg_24824 = r0_24732 * B_24815;
            float exp_arg_24825 = 0.0F - negate_arg_24824;
            float res_24826 = fpow32(2.7182817F, exp_arg_24825);
            float res_24827 = res_24823 * res_24826;
            float res_24809 = res_24827 + redout_25882;
            float redout_tmp_26456 = res_24809;
            
            redout_25882 = redout_tmp_26456;
        }
        res_24806 = redout_25882;
        
        float x_24828 = 1.0F - res_24796;
        float y_24829 = res_24764 * res_24806;
        float res_24830 = x_24828 / y_24829;
        
        ((float *) mem_26161)[i_25892] = res_24830;
        memmove(mem_26163 + i_25892 * 4, notional_mem_26159.mem + i_25892 * 4,
                (int32_t) sizeof(float));
        memmove(mem_26165 + i_25892 * 8, payments_mem_26158.mem + i_25892 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_26167 + i_25892 * 4, swap_term_mem_26157.mem + i_25892 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_24831 = res_24746 / res_24738;
    bool bounds_invalid_upwards_24832 = slt64(steps_24725, 1);
    bool valid_24833 = !bounds_invalid_upwards_24832;
    bool range_valid_c_24834;
    
    if (!valid_24833) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_24725,
                               " is invalid.",
                               "-> #0  cva.fut:60:56-67\n   #1  cva.fut:111:17-44\n   #2  cva.fut:100:1-165:21\n");
        if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
            return 1;
        return 1;
    }
    
    bool bounds_invalid_upwards_24836 = slt64(paths_24724, 0);
    bool valid_24837 = !bounds_invalid_upwards_24836;
    bool range_valid_c_24838;
    
    if (!valid_24837) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_24724,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:115:19-49\n   #4  cva.fut:100:1-165:21\n");
        if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
            return 1;
        return 1;
    }
    
    int64_t upper_bound_24841 = sub64(steps_24725, 1);
    float res_24842;
    
    res_24842 = futrts_sqrt32(dt_24747);
    
    int64_t binop_x_26231 = paths_24724 * steps_24725;
    int64_t bytes_26230 = 4 * binop_x_26231;
    
    if (mem_26232_cached_sizze_26491 < (size_t) bytes_26230) {
        mem_26232 = realloc(mem_26232, bytes_26230);
        mem_26232_cached_sizze_26491 = bytes_26230;
    }
    
    int64_t bytes_26243 = 4 * steps_24725;
    
    if (mem_26244_cached_sizze_26492 < (size_t) bytes_26243) {
        mem_26244 = realloc(mem_26244, bytes_26243);
        mem_26244_cached_sizze_26492 = bytes_26243;
    }
    if (mem_26258_cached_sizze_26493 < (size_t) bytes_26243) {
        mem_26258 = realloc(mem_26258, bytes_26243);
        mem_26258_cached_sizze_26493 = bytes_26243;
    }
    for (int64_t i_26457 = 0; i_26457 < steps_24725; i_26457++) {
        ((float *) mem_26258)[i_26457] = r0_24732;
    }
    for (int64_t i_25903 = 0; i_25903 < paths_24724; i_25903++) {
        int32_t res_24845 = sext_i64_i32(i_25903);
        int32_t x_24846 = lshr32(res_24845, 16);
        int32_t x_24847 = res_24845 ^ x_24846;
        int32_t x_24848 = mul32(73244475, x_24847);
        int32_t x_24849 = lshr32(x_24848, 16);
        int32_t x_24850 = x_24848 ^ x_24849;
        int32_t x_24851 = mul32(73244475, x_24850);
        int32_t x_24852 = lshr32(x_24851, 16);
        int32_t x_24853 = x_24851 ^ x_24852;
        int32_t unsign_arg_24854 = 777822902 ^ x_24853;
        int32_t unsign_arg_24855 = mul32(48271, unsign_arg_24854);
        int32_t unsign_arg_24856 = umod32(unsign_arg_24855, 2147483647);
        
        for (int64_t i_25899 = 0; i_25899 < steps_24725; i_25899++) {
            int32_t res_24859 = sext_i64_i32(i_25899);
            int32_t x_24860 = lshr32(res_24859, 16);
            int32_t x_24861 = res_24859 ^ x_24860;
            int32_t x_24862 = mul32(73244475, x_24861);
            int32_t x_24863 = lshr32(x_24862, 16);
            int32_t x_24864 = x_24862 ^ x_24863;
            int32_t x_24865 = mul32(73244475, x_24864);
            int32_t x_24866 = lshr32(x_24865, 16);
            int32_t x_24867 = x_24865 ^ x_24866;
            int32_t unsign_arg_24868 = unsign_arg_24856 ^ x_24867;
            int32_t unsign_arg_24869 = mul32(48271, unsign_arg_24868);
            int32_t unsign_arg_24870 = umod32(unsign_arg_24869, 2147483647);
            int32_t unsign_arg_24871 = mul32(48271, unsign_arg_24870);
            int32_t unsign_arg_24872 = umod32(unsign_arg_24871, 2147483647);
            float res_24873 = uitofp_i32_f32(unsign_arg_24870);
            float res_24874 = res_24873 / 2.1474836e9F;
            float res_24875 = uitofp_i32_f32(unsign_arg_24872);
            float res_24876 = res_24875 / 2.1474836e9F;
            float res_24877;
            
            res_24877 = futrts_log32(res_24874);
            
            float res_24878 = -2.0F * res_24877;
            float res_24879;
            
            res_24879 = futrts_sqrt32(res_24878);
            
            float res_24880 = 6.2831855F * res_24876;
            float res_24881;
            
            res_24881 = futrts_cos32(res_24880);
            
            float res_24882 = res_24879 * res_24881;
            
            ((float *) mem_26244)[i_25899] = res_24882;
        }
        memmove(mem_26232 + i_25903 * steps_24725 * 4, mem_26258 + 0,
                steps_24725 * (int32_t) sizeof(float));
        for (int64_t i_24885 = 0; i_24885 < upper_bound_24841; i_24885++) {
            bool y_24887 = slt64(i_24885, steps_24725);
            bool index_certs_24888;
            
            if (!y_24887) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_24885,
                              "] out of bounds for array of shape [",
                              steps_24725, "].",
                              "-> #0  cva.fut:71:97-104\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:100:1-165:21\n");
                if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                    return 1;
                return 1;
            }
            
            float shortstep_arg_24889 = ((float *) mem_26244)[i_24885];
            float shortstep_arg_24890 = ((float *) mem_26232)[i_25903 *
                                                              steps_24725 +
                                                              i_24885];
            float y_24891 = b_24730 - shortstep_arg_24890;
            float x_24892 = a_24729 * y_24891;
            float x_24893 = dt_24747 * x_24892;
            float x_24894 = res_24842 * shortstep_arg_24889;
            float y_24895 = sigma_24731 * x_24894;
            float delta_r_24896 = x_24893 + y_24895;
            float res_24897 = shortstep_arg_24890 + delta_r_24896;
            int64_t i_24898 = add64(1, i_24885);
            bool x_24899 = sle64(0, i_24898);
            bool y_24900 = slt64(i_24898, steps_24725);
            bool bounds_check_24901 = x_24899 && y_24900;
            bool index_certs_24902;
            
            if (!bounds_check_24901) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_24898,
                              "] out of bounds for array of shape [",
                              steps_24725, "].",
                              "-> #0  cva.fut:71:58-105\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:100:1-165:21\n");
                if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                    return 1;
                return 1;
            }
            ((float *) mem_26232)[i_25903 * steps_24725 + i_24898] = res_24897;
        }
    }
    
    int64_t flat_dim_24906 = n_24721 * paths_24724;
    float res_24910 = sitofp_i64_f32(paths_24724);
    int64_t bytes_26285 = 4 * steps_24725;
    
    if (mem_26286_cached_sizze_26494 < (size_t) bytes_26285) {
        mem_26286 = realloc(mem_26286, bytes_26285);
        mem_26286_cached_sizze_26494 = bytes_26285;
    }
    
    int64_t binop_x_26295 = n_24721 * paths_24724;
    int64_t bytes_26294 = 4 * binop_x_26295;
    
    if (mem_26296_cached_sizze_26495 < (size_t) bytes_26294) {
        mem_26296 = realloc(mem_26296, bytes_26294);
        mem_26296_cached_sizze_26495 = bytes_26294;
    }
    if (mem_26308_cached_sizze_26496 < (size_t) bytes_26160) {
        mem_26308 = realloc(mem_26308, bytes_26160);
        mem_26308_cached_sizze_26496 = bytes_26160;
    }
    
    int64_t bytes_26317 = 8 * flat_dim_24906;
    
    if (mem_26318_cached_sizze_26497 < (size_t) bytes_26317) {
        mem_26318 = realloc(mem_26318, bytes_26317);
        mem_26318_cached_sizze_26497 = bytes_26317;
    }
    if (mem_26320_cached_sizze_26498 < (size_t) bytes_26317) {
        mem_26320 = realloc(mem_26320, bytes_26317);
        mem_26320_cached_sizze_26498 = bytes_26317;
    }
    
    float res_25163;
    float redout_26006 = 0.0F;
    
    for (int64_t i_26008 = 0; i_26008 < steps_24725; i_26008++) {
        int64_t index_primexp_26148 = add64(1, i_26008);
        float res_25170 = sitofp_i64_f32(index_primexp_26148);
        float res_25171 = res_25170 / sims_per_year_24831;
        
        for (int64_t i_25907 = 0; i_25907 < paths_24724; i_25907++) {
            float x_25173 = ((float *) mem_26232)[i_25907 * steps_24725 +
                                                  i_26008];
            
            for (int64_t i_26464 = 0; i_26464 < n_24721; i_26464++) {
                ((float *) mem_26308)[i_26464] = x_25173;
            }
            memmove(mem_26296 + i_25907 * n_24721 * 4, mem_26308 + 0, n_24721 *
                    (int32_t) sizeof(float));
        }
        
        int64_t discard_25917;
        int64_t scanacc_25911 = 0;
        
        for (int64_t i_25914 = 0; i_25914 < flat_dim_24906; i_25914++) {
            int64_t binop_x_26023 = squot64(i_25914, n_24721);
            int64_t binop_y_26024 = n_24721 * binop_x_26023;
            int64_t new_index_26025 = i_25914 - binop_y_26024;
            int64_t x_25180 = ((int64_t *) mem_26165)[new_index_26025];
            float x_25181 = ((float *) mem_26167)[new_index_26025];
            float x_25182 = res_25171 / x_25181;
            float ceil_arg_25183 = x_25182 - 1.0F;
            float res_25184;
            
            res_25184 = futrts_ceil32(ceil_arg_25183);
            
            int64_t res_25185 = fptosi_f32_i64(res_25184);
            int64_t max_arg_25186 = sub64(x_25180, res_25185);
            int64_t res_25187 = smax64(0, max_arg_25186);
            bool cond_25188 = res_25187 == 0;
            int64_t res_25189;
            
            if (cond_25188) {
                res_25189 = 1;
            } else {
                res_25189 = res_25187;
            }
            
            int64_t res_25179 = add64(res_25189, scanacc_25911);
            
            ((int64_t *) mem_26318)[i_25914] = res_25179;
            ((int64_t *) mem_26320)[i_25914] = res_25189;
            
            int64_t scanacc_tmp_26465 = res_25179;
            
            scanacc_25911 = scanacc_tmp_26465;
        }
        discard_25917 = scanacc_25911;
        
        int64_t res_25191;
        int64_t redout_25918 = 0;
        
        for (int64_t i_25919 = 0; i_25919 < flat_dim_24906; i_25919++) {
            int64_t x_25195 = ((int64_t *) mem_26320)[i_25919];
            int64_t res_25194 = add64(x_25195, redout_25918);
            int64_t redout_tmp_26468 = res_25194;
            
            redout_25918 = redout_tmp_26468;
        }
        res_25191 = redout_25918;
        
        bool bounds_invalid_upwards_25196 = slt64(res_25191, 0);
        bool valid_25197 = !bounds_invalid_upwards_25196;
        bool range_valid_c_25198;
        
        if (!valid_25197) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", res_25191,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:132:34-93\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:126:18-137:50\n   #9  cva.fut:100:1-165:21\n");
            if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                return 1;
            return 1;
        }
        
        int64_t bytes_26345 = 8 * res_25191;
        
        if (mem_26346_cached_sizze_26499 < (size_t) bytes_26345) {
            mem_26346 = realloc(mem_26346, bytes_26345);
            mem_26346_cached_sizze_26499 = bytes_26345;
        }
        for (int64_t i_26469 = 0; i_26469 < res_25191; i_26469++) {
            ((int64_t *) mem_26346)[i_26469] = 0;
        }
        for (int64_t iter_25920 = 0; iter_25920 < flat_dim_24906;
             iter_25920++) {
            int64_t i_p_o_26030 = add64(-1, iter_25920);
            int64_t rot_i_26031 = smod64(i_p_o_26030, flat_dim_24906);
            int64_t pixel_25923 = ((int64_t *) mem_26318)[rot_i_26031];
            bool cond_25206 = iter_25920 == 0;
            int64_t res_25207;
            
            if (cond_25206) {
                res_25207 = 0;
            } else {
                res_25207 = pixel_25923;
            }
            
            bool less_than_zzero_25924 = slt64(res_25207, 0);
            bool greater_than_sizze_25925 = sle64(res_25191, res_25207);
            bool outside_bounds_dim_25926 = less_than_zzero_25924 ||
                 greater_than_sizze_25925;
            
            if (!outside_bounds_dim_25926) {
                int64_t read_hist_25928 = ((int64_t *) mem_26346)[res_25207];
                int64_t res_25203 = smax64(iter_25920, read_hist_25928);
                
                ((int64_t *) mem_26346)[res_25207] = res_25203;
            }
        }
        if (mem_26360_cached_sizze_26500 < (size_t) bytes_26345) {
            mem_26360 = realloc(mem_26360, bytes_26345);
            mem_26360_cached_sizze_26500 = bytes_26345;
        }
        
        int64_t discard_25941;
        int64_t scanacc_25934 = 0;
        
        for (int64_t i_25937 = 0; i_25937 < res_25191; i_25937++) {
            int64_t x_25217 = ((int64_t *) mem_26346)[i_25937];
            bool res_25218 = slt64(0, x_25217);
            int64_t res_25215;
            
            if (res_25218) {
                res_25215 = x_25217;
            } else {
                int64_t res_25216 = add64(x_25217, scanacc_25934);
                
                res_25215 = res_25216;
            }
            ((int64_t *) mem_26360)[i_25937] = res_25215;
            
            int64_t scanacc_tmp_26471 = res_25215;
            
            scanacc_25934 = scanacc_tmp_26471;
        }
        discard_25941 = scanacc_25934;
        
        int64_t bytes_26373 = 4 * res_25191;
        
        if (mem_26374_cached_sizze_26501 < (size_t) bytes_26373) {
            mem_26374 = realloc(mem_26374, bytes_26373);
            mem_26374_cached_sizze_26501 = bytes_26373;
        }
        if (mem_26376_cached_sizze_26502 < (size_t) res_25191) {
            mem_26376 = realloc(mem_26376, res_25191);
            mem_26376_cached_sizze_26502 = res_25191;
        }
        
        int64_t inpacc_25227;
        float inpacc_25229;
        int64_t inpacc_25235;
        float inpacc_25237;
        
        inpacc_25235 = 0;
        inpacc_25237 = 0.0F;
        for (int64_t i_25980 = 0; i_25980 < res_25191; i_25980++) {
            int64_t x_26045 = ((int64_t *) mem_26360)[i_25980];
            int64_t i_p_o_26047 = add64(-1, i_25980);
            int64_t rot_i_26048 = smod64(i_p_o_26047, res_25191);
            int64_t x_26049 = ((int64_t *) mem_26360)[rot_i_26048];
            bool res_26050 = x_26045 == x_26049;
            bool res_26051 = !res_26050;
            int64_t res_25271;
            
            if (res_26051) {
                res_25271 = 1;
            } else {
                int64_t res_25272 = add64(1, inpacc_25235);
                
                res_25271 = res_25272;
            }
            
            int64_t res_25286;
            
            if (res_26051) {
                res_25286 = 1;
            } else {
                int64_t res_25287 = add64(1, inpacc_25235);
                
                res_25286 = res_25287;
            }
            
            int64_t res_25288 = sub64(res_25286, 1);
            bool x_26066 = sle64(0, x_26045);
            bool y_26067 = slt64(x_26045, flat_dim_24906);
            bool bounds_check_26068 = x_26066 && y_26067;
            bool index_certs_26069;
            
            if (!bounds_check_26068) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", x_26045,
                              "] out of bounds for array of shape [",
                              flat_dim_24906, "].",
                              "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:132:34-93\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:126:18-137:50\n   #9  cva.fut:100:1-165:21\n");
                if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                    return 1;
                return 1;
            }
            
            int64_t new_index_26070 = squot64(x_26045, n_24721);
            int64_t binop_y_26071 = n_24721 * new_index_26070;
            int64_t new_index_26072 = x_26045 - binop_y_26071;
            float lifted_0_get_arg_26073 =
                  ((float *) mem_26296)[new_index_26070 * n_24721 +
                                        new_index_26072];
            float lifted_0_get_arg_26074 =
                  ((float *) mem_26161)[new_index_26072];
            float lifted_0_get_arg_26075 =
                  ((float *) mem_26163)[new_index_26072];
            int64_t lifted_0_get_arg_26076 =
                    ((int64_t *) mem_26165)[new_index_26072];
            float lifted_0_get_arg_26077 =
                  ((float *) mem_26167)[new_index_26072];
            float x_26078 = res_25171 / lifted_0_get_arg_26077;
            float ceil_arg_26079 = x_26078 - 1.0F;
            float res_26080;
            
            res_26080 = futrts_ceil32(ceil_arg_26079);
            
            int64_t res_26081 = fptosi_f32_i64(res_26080);
            int64_t max_arg_26082 = sub64(lifted_0_get_arg_26076, res_26081);
            int64_t res_26083 = smax64(0, max_arg_26082);
            bool cond_26084 = res_26083 == 0;
            float res_26085;
            
            if (cond_26084) {
                res_26085 = 0.0F;
            } else {
                float res_26086;
                
                res_26086 = futrts_ceil32(x_26078);
                
                float start_26087 = lifted_0_get_arg_26077 * res_26086;
                float res_26088;
                
                res_26088 = futrts_ceil32(ceil_arg_26079);
                
                int64_t res_26089 = fptosi_f32_i64(res_26088);
                int64_t max_arg_26090 = sub64(lifted_0_get_arg_26076,
                                              res_26089);
                int64_t res_26091 = smax64(0, max_arg_26090);
                int64_t sizze_26092 = sub64(res_26091, 1);
                bool cond_26093 = res_25288 == 0;
                float res_26094;
                
                if (cond_26093) {
                    res_26094 = 1.0F;
                } else {
                    res_26094 = 0.0F;
                }
                
                bool cond_26095 = slt64(0, res_25288);
                float res_26096;
                
                if (cond_26095) {
                    float y_26097 = lifted_0_get_arg_26074 *
                          lifted_0_get_arg_26077;
                    float res_26098 = res_26094 - y_26097;
                    
                    res_26096 = res_26098;
                } else {
                    res_26096 = res_26094;
                }
                
                bool cond_26099 = res_25288 == sizze_26092;
                float res_26100;
                
                if (cond_26099) {
                    float res_26101 = res_26096 - 1.0F;
                    
                    res_26100 = res_26101;
                } else {
                    res_26100 = res_26096;
                }
                
                float res_26102 = lifted_0_get_arg_26075 * res_26100;
                float res_26103 = sitofp_i64_f32(res_25288);
                float y_26104 = lifted_0_get_arg_26077 * res_26103;
                float bondprice_arg_26105 = start_26087 + y_26104;
                float y_26106 = bondprice_arg_26105 - res_25171;
                float negate_arg_26107 = a_24729 * y_26106;
                float exp_arg_26108 = 0.0F - negate_arg_26107;
                float res_26109 = fpow32(2.7182817F, exp_arg_26108);
                float x_26110 = 1.0F - res_26109;
                float B_26111 = x_26110 / a_24729;
                float x_26112 = B_26111 - bondprice_arg_26105;
                float x_26113 = res_25171 + x_26112;
                float x_26119 = y_24753 * x_26113;
                float A1_26120 = x_26119 / x_24749;
                float y_26121 = fpow32(B_26111, 2.0F);
                float x_26122 = x_24751 * y_26121;
                float A2_26124 = x_26122 / y_24754;
                float exp_arg_26125 = A1_26120 - A2_26124;
                float res_26126 = fpow32(2.7182817F, exp_arg_26125);
                float negate_arg_26127 = lifted_0_get_arg_26073 * B_26111;
                float exp_arg_26128 = 0.0F - negate_arg_26127;
                float res_26129 = fpow32(2.7182817F, exp_arg_26128);
                float res_26130 = res_26126 * res_26129;
                float res_26131 = res_26102 * res_26130;
                
                res_26085 = res_26131;
            }
            
            float res_25364;
            
            if (res_26051) {
                res_25364 = res_26085;
            } else {
                float res_25365 = inpacc_25237 + res_26085;
                
                res_25364 = res_25365;
            }
            
            float res_25367;
            
            if (res_26051) {
                res_25367 = res_26085;
            } else {
                float res_25368 = inpacc_25237 + res_26085;
                
                res_25367 = res_25368;
            }
            ((float *) mem_26374)[i_25980] = res_25364;
            ((bool *) mem_26376)[i_25980] = res_26051;
            
            int64_t inpacc_tmp_26473 = res_25271;
            float inpacc_tmp_26474 = res_25367;
            
            inpacc_25235 = inpacc_tmp_26473;
            inpacc_25237 = inpacc_tmp_26474;
        }
        inpacc_25227 = inpacc_25235;
        inpacc_25229 = inpacc_25237;
        if (mem_26402_cached_sizze_26503 < (size_t) bytes_26345) {
            mem_26402 = realloc(mem_26402, bytes_26345);
            mem_26402_cached_sizze_26503 = bytes_26345;
        }
        
        int64_t discard_25989;
        int64_t scanacc_25985 = 0;
        
        for (int64_t i_25987 = 0; i_25987 < res_25191; i_25987++) {
            int64_t i_p_o_26142 = add64(1, i_25987);
            int64_t rot_i_26143 = smod64(i_p_o_26142, res_25191);
            bool x_25381 = ((bool *) mem_26376)[rot_i_26143];
            int64_t res_25382 = btoi_bool_i64(x_25381);
            int64_t res_25380 = add64(res_25382, scanacc_25985);
            
            ((int64_t *) mem_26402)[i_25987] = res_25380;
            
            int64_t scanacc_tmp_26477 = res_25380;
            
            scanacc_25985 = scanacc_tmp_26477;
        }
        discard_25989 = scanacc_25985;
        
        bool cond_25383 = slt64(0, res_25191);
        int64_t num_segments_25384;
        
        if (cond_25383) {
            int64_t i_25385 = sub64(res_25191, 1);
            bool x_25386 = sle64(0, i_25385);
            bool y_25387 = slt64(i_25385, res_25191);
            bool bounds_check_25388 = x_25386 && y_25387;
            bool index_certs_25389;
            
            if (!bounds_check_25388) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25385,
                              "] out of bounds for array of shape [", res_25191,
                              "].",
                              "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva.fut:132:34-93\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:126:18-137:50\n   #8  cva.fut:100:1-165:21\n");
                if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                    return 1;
                return 1;
            }
            
            int64_t res_25390 = ((int64_t *) mem_26402)[i_25385];
            
            num_segments_25384 = res_25390;
        } else {
            num_segments_25384 = 0;
        }
        
        bool bounds_invalid_upwards_25391 = slt64(num_segments_25384, 0);
        bool valid_25392 = !bounds_invalid_upwards_25391;
        bool range_valid_c_25393;
        
        if (!valid_25392) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", num_segments_25384,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:132:34-93\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:126:18-137:50\n   #9  cva.fut:100:1-165:21\n");
            if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                return 1;
            return 1;
        }
        
        int64_t bytes_26415 = 4 * num_segments_25384;
        
        if (mem_26416_cached_sizze_26504 < (size_t) bytes_26415) {
            mem_26416 = realloc(mem_26416, bytes_26415);
            mem_26416_cached_sizze_26504 = bytes_26415;
        }
        for (int64_t i_26479 = 0; i_26479 < num_segments_25384; i_26479++) {
            ((float *) mem_26416)[i_26479] = 0.0F;
        }
        for (int64_t write_iter_25990 = 0; write_iter_25990 < res_25191;
             write_iter_25990++) {
            int64_t write_iv_25992 = ((int64_t *) mem_26402)[write_iter_25990];
            int64_t i_p_o_26145 = add64(1, write_iter_25990);
            int64_t rot_i_26146 = smod64(i_p_o_26145, res_25191);
            bool write_iv_25993 = ((bool *) mem_26376)[rot_i_26146];
            int64_t res_25399;
            
            if (write_iv_25993) {
                int64_t res_25400 = sub64(write_iv_25992, 1);
                
                res_25399 = res_25400;
            } else {
                res_25399 = -1;
            }
            
            bool less_than_zzero_25995 = slt64(res_25399, 0);
            bool greater_than_sizze_25996 = sle64(num_segments_25384,
                                                  res_25399);
            bool outside_bounds_dim_25997 = less_than_zzero_25995 ||
                 greater_than_sizze_25996;
            
            if (!outside_bounds_dim_25997) {
                memmove(mem_26416 + res_25399 * 4, mem_26374 +
                        write_iter_25990 * 4, (int32_t) sizeof(float));
            }
        }
        
        bool dim_match_25401 = flat_dim_24906 == num_segments_25384;
        bool empty_or_match_cert_25402;
        
        if (!dim_match_25401) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Value of (core language) shape (",
                                   num_segments_25384,
                                   ") cannot match shape of type `[",
                                   flat_dim_24906, "]b`.",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva.fut:132:34-93\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:126:18-137:50\n   #5  cva.fut:100:1-165:21\n");
            if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
                return 1;
            return 1;
        }
        
        float res_25404;
        float redout_26003 = 0.0F;
        
        for (int64_t i_26004 = 0; i_26004 < paths_24724; i_26004++) {
            int64_t binop_x_26149 = n_24721 * i_26004;
            float res_25409;
            float redout_26001 = 0.0F;
            
            for (int64_t i_26002 = 0; i_26002 < n_24721; i_26002++) {
                int64_t new_index_26150 = i_26002 + binop_x_26149;
                float x_25413 = ((float *) mem_26416)[new_index_26150];
                float res_25412 = x_25413 + redout_26001;
                float redout_tmp_26482 = res_25412;
                
                redout_26001 = redout_tmp_26482;
            }
            res_25409 = redout_26001;
            
            float res_25414 = fmax32(0.0F, res_25409);
            float res_25407 = res_25414 + redout_26003;
            float redout_tmp_26481 = res_25407;
            
            redout_26003 = redout_tmp_26481;
        }
        res_25404 = redout_26003;
        
        float res_25415 = res_25404 / res_24910;
        float negate_arg_25416 = a_24729 * res_25171;
        float exp_arg_25417 = 0.0F - negate_arg_25416;
        float res_25418 = fpow32(2.7182817F, exp_arg_25417);
        float x_25419 = 1.0F - res_25418;
        float B_25420 = x_25419 / a_24729;
        float x_25421 = B_25420 - res_25171;
        float x_25422 = y_24753 * x_25421;
        float A1_25423 = x_25422 / x_24749;
        float y_25424 = fpow32(B_25420, 2.0F);
        float x_25425 = x_24751 * y_25424;
        float A2_25426 = x_25425 / y_24754;
        float exp_arg_25427 = A1_25423 - A2_25426;
        float res_25428 = fpow32(2.7182817F, exp_arg_25427);
        float negate_arg_25429 = 5.0e-2F * B_25420;
        float exp_arg_25430 = 0.0F - negate_arg_25429;
        float res_25431 = fpow32(2.7182817F, exp_arg_25430);
        float res_25432 = res_25428 * res_25431;
        float res_25433 = res_25415 * res_25432;
        float res_25167 = res_25433 + redout_26006;
        
        ((float *) mem_26286)[i_26008] = res_25415;
        
        float redout_tmp_26461 = res_25167;
        
        redout_26006 = redout_tmp_26461;
    }
    res_25163 = redout_26006;
    
    float CVA_25436 = 6.0e-3F * res_25163;
    struct memblock mem_26435;
    
    mem_26435.references = NULL;
    if (memblock_alloc(ctx, &mem_26435, bytes_26285, "mem_26435")) {
        err = 1;
        goto cleanup;
    }
    memmove(mem_26435.mem + 0, mem_26286 + 0, steps_24725 *
            (int32_t) sizeof(float));
    out_arrsizze_26449 = steps_24725;
    if (memblock_set(ctx, &out_mem_26448, &mem_26435, "mem_26435") != 0)
        return 1;
    scalar_out_26447 = CVA_25436;
    *out_scalar_out_26483 = scalar_out_26447;
    (*out_mem_p_26484).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_26484, &out_mem_26448, "out_mem_26448") !=
        0)
        return 1;
    *out_out_arrsizze_26485 = out_arrsizze_26449;
    if (memblock_unref(ctx, &mem_26435, "mem_26435") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_26448, "out_mem_26448") != 0)
        return 1;
    
  cleanup:
    { }
    free(mem_26161);
    free(mem_26163);
    free(mem_26165);
    free(mem_26167);
    free(mem_26197);
    free(mem_26232);
    free(mem_26244);
    free(mem_26258);
    free(mem_26286);
    free(mem_26296);
    free(mem_26308);
    free(mem_26318);
    free(mem_26320);
    free(mem_26346);
    free(mem_26360);
    free(mem_26374);
    free(mem_26376);
    free(mem_26402);
    free(mem_26416);
    return err;
}
static int futrts_test(struct futhark_context *ctx, float *out_scalar_out_26505,
                       int64_t paths_25438, int64_t steps_25439)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_26158_cached_sizze_26506 = 0;
    char *mem_26158 = NULL;
    size_t mem_26160_cached_sizze_26507 = 0;
    char *mem_26160 = NULL;
    size_t mem_26162_cached_sizze_26508 = 0;
    char *mem_26162 = NULL;
    size_t mem_26164_cached_sizze_26509 = 0;
    char *mem_26164 = NULL;
    size_t mem_26166_cached_sizze_26510 = 0;
    char *mem_26166 = NULL;
    size_t mem_26168_cached_sizze_26511 = 0;
    char *mem_26168 = NULL;
    size_t mem_26170_cached_sizze_26512 = 0;
    char *mem_26170 = NULL;
    size_t mem_26208_cached_sizze_26513 = 0;
    char *mem_26208 = NULL;
    size_t mem_26243_cached_sizze_26514 = 0;
    char *mem_26243 = NULL;
    size_t mem_26255_cached_sizze_26515 = 0;
    char *mem_26255 = NULL;
    size_t mem_26269_cached_sizze_26516 = 0;
    char *mem_26269 = NULL;
    size_t mem_26298_cached_sizze_26517 = 0;
    char *mem_26298 = NULL;
    size_t mem_26313_cached_sizze_26518 = 0;
    char *mem_26313 = NULL;
    size_t mem_26323_cached_sizze_26519 = 0;
    char *mem_26323 = NULL;
    size_t mem_26325_cached_sizze_26520 = 0;
    char *mem_26325 = NULL;
    size_t mem_26351_cached_sizze_26521 = 0;
    char *mem_26351 = NULL;
    size_t mem_26365_cached_sizze_26522 = 0;
    char *mem_26365 = NULL;
    size_t mem_26379_cached_sizze_26523 = 0;
    char *mem_26379 = NULL;
    size_t mem_26381_cached_sizze_26524 = 0;
    char *mem_26381 = NULL;
    size_t mem_26407_cached_sizze_26525 = 0;
    char *mem_26407 = NULL;
    size_t mem_26421_cached_sizze_26526 = 0;
    char *mem_26421 = NULL;
    float scalar_out_26447;
    
    if (mem_26158_cached_sizze_26506 < (size_t) 36) {
        mem_26158 = realloc(mem_26158, 36);
        mem_26158_cached_sizze_26506 = 36;
    }
    
    struct memblock testzistatic_array_26448 = ctx->testzistatic_array_26448;
    
    memmove(mem_26158 + 0, testzistatic_array_26448.mem + 0, 9 *
            (int32_t) sizeof(float));
    if (mem_26160_cached_sizze_26507 < (size_t) 72) {
        mem_26160 = realloc(mem_26160, 72);
        mem_26160_cached_sizze_26507 = 72;
    }
    
    struct memblock testzistatic_array_26449 = ctx->testzistatic_array_26449;
    
    memmove(mem_26160 + 0, testzistatic_array_26449.mem + 0, 9 *
            (int32_t) sizeof(int64_t));
    if (mem_26162_cached_sizze_26508 < (size_t) 36) {
        mem_26162 = realloc(mem_26162, 36);
        mem_26162_cached_sizze_26508 = 36;
    }
    
    struct memblock testzistatic_array_26450 = ctx->testzistatic_array_26450;
    
    memmove(mem_26162 + 0, testzistatic_array_26450.mem + 0, 9 *
            (int32_t) sizeof(float));
    
    float res_25443;
    float redout_25876 = -INFINITY;
    
    for (int32_t i_26010 = 0; i_26010 < 9; i_26010++) {
        int64_t i_25877 = sext_i32_i64(i_26010);
        float x_25447 = ((float *) mem_26162)[i_25877];
        int64_t x_25448 = ((int64_t *) mem_26160)[i_25877];
        float res_25449 = sitofp_i64_f32(x_25448);
        float res_25450 = x_25447 * res_25449;
        float res_25446 = fmax32(res_25450, redout_25876);
        float redout_tmp_26451 = res_25446;
        
        redout_25876 = redout_tmp_26451;
    }
    res_25443 = redout_25876;
    
    float res_25451 = sitofp_i64_f32(steps_25439);
    float dt_25452 = res_25443 / res_25451;
    
    if (mem_26164_cached_sizze_26509 < (size_t) 36) {
        mem_26164 = realloc(mem_26164, 36);
        mem_26164_cached_sizze_26509 = 36;
    }
    if (mem_26166_cached_sizze_26510 < (size_t) 36) {
        mem_26166 = realloc(mem_26166, 36);
        mem_26166_cached_sizze_26510 = 36;
    }
    if (mem_26168_cached_sizze_26511 < (size_t) 72) {
        mem_26168 = realloc(mem_26168, 72);
        mem_26168_cached_sizze_26511 = 72;
    }
    if (mem_26170_cached_sizze_26512 < (size_t) 36) {
        mem_26170 = realloc(mem_26170, 36);
        mem_26170_cached_sizze_26512 = 36;
    }
    for (int32_t i_26019 = 0; i_26019 < 9; i_26019++) {
        int64_t i_25892 = sext_i32_i64(i_26019);
        bool x_25459 = sle64(0, i_25892);
        bool y_25460 = slt64(i_25892, 9);
        bool bounds_check_25461 = x_25459 && y_25460;
        bool index_certs_25462;
        
        if (!bounds_check_25461) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", i_25892,
                                   "] out of bounds for array of shape [", 9,
                                   "].",
                                   "-> #0  cva.fut:106:15-26\n   #1  cva.fut:105:17-109:85\n   #2  cva.fut:173:3-120\n   #3  cva.fut:172:1-173:128\n");
            return 1;
        }
        
        float res_25463 = ((float *) mem_26162)[i_25892];
        int64_t res_25464 = ((int64_t *) mem_26160)[i_25892];
        int64_t range_end_25466 = sub64(res_25464, 1);
        bool bounds_invalid_upwards_25467 = slt64(range_end_25466, 0);
        bool valid_25468 = !bounds_invalid_upwards_25467;
        bool range_valid_c_25469;
        
        if (!valid_25468) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "...", range_end_25466,
                          " is invalid.",
                          "-> #0  cva.fut:54:29-52\n   #1  cva.fut:95:25-65\n   #2  cva.fut:109:16-62\n   #3  cva.fut:105:17-109:85\n   #4  cva.fut:173:3-120\n   #5  cva.fut:172:1-173:128\n");
            return 1;
        }
        
        int64_t bytes_26207 = 4 * res_25464;
        
        if (mem_26208_cached_sizze_26513 < (size_t) bytes_26207) {
            mem_26208 = realloc(mem_26208, bytes_26207);
            mem_26208_cached_sizze_26513 = bytes_26207;
        }
        for (int64_t i_25880 = 0; i_25880 < res_25464; i_25880++) {
            float res_25473 = sitofp_i64_f32(i_25880);
            float res_25474 = res_25463 * res_25473;
            
            ((float *) mem_26208)[i_25880] = res_25474;
        }
        
        bool y_25475 = slt64(0, res_25464);
        bool index_certs_25476;
        
        if (!y_25475) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 0,
                                   "] out of bounds for array of shape [",
                                   res_25464, "].",
                                   "-> #0  cva.fut:96:47-70\n   #1  cva.fut:109:16-62\n   #2  cva.fut:105:17-109:85\n   #3  cva.fut:173:3-120\n   #4  cva.fut:172:1-173:128\n");
            return 1;
        }
        
        float binop_y_25477 = sitofp_i64_f32(range_end_25466);
        float index_primexp_25478 = res_25463 * binop_y_25477;
        float negate_arg_25479 = 1.0e-2F * index_primexp_25478;
        float exp_arg_25480 = 0.0F - negate_arg_25479;
        float res_25481 = fpow32(2.7182817F, exp_arg_25480);
        float x_25482 = 1.0F - res_25481;
        float B_25483 = x_25482 / 1.0e-2F;
        float x_25484 = B_25483 - index_primexp_25478;
        float x_25485 = 4.4999997e-6F * x_25484;
        float A1_25486 = x_25485 / 1.0e-4F;
        float y_25487 = fpow32(B_25483, 2.0F);
        float x_25488 = 1.0000001e-6F * y_25487;
        float A2_25489 = x_25488 / 4.0e-2F;
        float exp_arg_25490 = A1_25486 - A2_25489;
        float res_25491 = fpow32(2.7182817F, exp_arg_25490);
        float negate_arg_25492 = 5.0e-2F * B_25483;
        float exp_arg_25493 = 0.0F - negate_arg_25492;
        float res_25494 = fpow32(2.7182817F, exp_arg_25493);
        float res_25495 = res_25491 * res_25494;
        bool empty_slice_25496 = range_end_25466 == 0;
        bool zzero_leq_i_p_m_t_s_25497 = sle64(0, range_end_25466);
        bool i_p_m_t_s_leq_w_25498 = slt64(range_end_25466, res_25464);
        bool i_lte_j_25499 = sle64(1, res_25464);
        bool y_25500 = zzero_leq_i_p_m_t_s_25497 && i_p_m_t_s_leq_w_25498;
        bool y_25501 = i_lte_j_25499 && y_25500;
        bool ok_or_empty_25502 = empty_slice_25496 || y_25501;
        bool index_certs_25503;
        
        if (!ok_or_empty_25502) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Index [", 1,
                                   ":] out of bounds for array of shape [",
                                   res_25464, "].",
                                   "-> #0  cva.fut:97:74-90\n   #1  cva.fut:109:16-62\n   #2  cva.fut:105:17-109:85\n   #3  cva.fut:173:3-120\n   #4  cva.fut:172:1-173:128\n");
            return 1;
        }
        
        float res_25505;
        float redout_25882 = 0.0F;
        
        for (int64_t i_25883 = 0; i_25883 < range_end_25466; i_25883++) {
            int64_t slice_26013 = 1 + i_25883;
            float x_25509 = ((float *) mem_26208)[slice_26013];
            float negate_arg_25510 = 1.0e-2F * x_25509;
            float exp_arg_25511 = 0.0F - negate_arg_25510;
            float res_25512 = fpow32(2.7182817F, exp_arg_25511);
            float x_25513 = 1.0F - res_25512;
            float B_25514 = x_25513 / 1.0e-2F;
            float x_25515 = B_25514 - x_25509;
            float x_25516 = 4.4999997e-6F * x_25515;
            float A1_25517 = x_25516 / 1.0e-4F;
            float y_25518 = fpow32(B_25514, 2.0F);
            float x_25519 = 1.0000001e-6F * y_25518;
            float A2_25520 = x_25519 / 4.0e-2F;
            float exp_arg_25521 = A1_25517 - A2_25520;
            float res_25522 = fpow32(2.7182817F, exp_arg_25521);
            float negate_arg_25523 = 5.0e-2F * B_25514;
            float exp_arg_25524 = 0.0F - negate_arg_25523;
            float res_25525 = fpow32(2.7182817F, exp_arg_25524);
            float res_25526 = res_25522 * res_25525;
            float res_25508 = res_25526 + redout_25882;
            float redout_tmp_26457 = res_25508;
            
            redout_25882 = redout_tmp_26457;
        }
        res_25505 = redout_25882;
        
        float x_25527 = 1.0F - res_25495;
        float y_25528 = res_25463 * res_25505;
        float res_25529 = x_25527 / y_25528;
        
        ((float *) mem_26164)[i_25892] = res_25529;
        memmove(mem_26166 + i_25892 * 4, mem_26158 + i_25892 * 4,
                (int32_t) sizeof(float));
        memmove(mem_26168 + i_25892 * 8, mem_26160 + i_25892 * 8,
                (int32_t) sizeof(int64_t));
        memmove(mem_26170 + i_25892 * 4, mem_26162 + i_25892 * 4,
                (int32_t) sizeof(float));
    }
    
    float sims_per_year_25530 = res_25451 / res_25443;
    bool bounds_invalid_upwards_25531 = slt64(steps_25439, 1);
    bool valid_25532 = !bounds_invalid_upwards_25531;
    bool range_valid_c_25533;
    
    if (!valid_25532) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..", 2, "...", steps_25439,
                               " is invalid.",
                               "-> #0  cva.fut:60:56-67\n   #1  cva.fut:111:17-44\n   #2  cva.fut:173:3-120\n   #3  cva.fut:172:1-173:128\n");
        return 1;
    }
    
    bool bounds_invalid_upwards_25535 = slt64(paths_25438, 0);
    bool valid_25536 = !bounds_invalid_upwards_25535;
    bool range_valid_c_25537;
    
    if (!valid_25536) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", paths_25438,
                               " is invalid.",
                               "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:126:11-16\n   #2  lib/github.com/diku-dk/cpprandom/random.fut:174:8-56\n   #3  cva.fut:115:19-49\n   #4  cva.fut:173:3-120\n   #5  cva.fut:172:1-173:128\n");
        return 1;
    }
    
    int64_t upper_bound_25540 = sub64(steps_25439, 1);
    float res_25541;
    
    res_25541 = futrts_sqrt32(dt_25452);
    
    int64_t binop_x_26242 = paths_25438 * steps_25439;
    int64_t bytes_26241 = 4 * binop_x_26242;
    
    if (mem_26243_cached_sizze_26514 < (size_t) bytes_26241) {
        mem_26243 = realloc(mem_26243, bytes_26241);
        mem_26243_cached_sizze_26514 = bytes_26241;
    }
    
    int64_t bytes_26254 = 4 * steps_25439;
    
    if (mem_26255_cached_sizze_26515 < (size_t) bytes_26254) {
        mem_26255 = realloc(mem_26255, bytes_26254);
        mem_26255_cached_sizze_26515 = bytes_26254;
    }
    if (mem_26269_cached_sizze_26516 < (size_t) bytes_26254) {
        mem_26269 = realloc(mem_26269, bytes_26254);
        mem_26269_cached_sizze_26516 = bytes_26254;
    }
    for (int64_t i_26458 = 0; i_26458 < steps_25439; i_26458++) {
        ((float *) mem_26269)[i_26458] = 5.0e-2F;
    }
    for (int64_t i_25903 = 0; i_25903 < paths_25438; i_25903++) {
        int32_t res_25544 = sext_i64_i32(i_25903);
        int32_t x_25545 = lshr32(res_25544, 16);
        int32_t x_25546 = res_25544 ^ x_25545;
        int32_t x_25547 = mul32(73244475, x_25546);
        int32_t x_25548 = lshr32(x_25547, 16);
        int32_t x_25549 = x_25547 ^ x_25548;
        int32_t x_25550 = mul32(73244475, x_25549);
        int32_t x_25551 = lshr32(x_25550, 16);
        int32_t x_25552 = x_25550 ^ x_25551;
        int32_t unsign_arg_25553 = 777822902 ^ x_25552;
        int32_t unsign_arg_25554 = mul32(48271, unsign_arg_25553);
        int32_t unsign_arg_25555 = umod32(unsign_arg_25554, 2147483647);
        
        for (int64_t i_25899 = 0; i_25899 < steps_25439; i_25899++) {
            int32_t res_25558 = sext_i64_i32(i_25899);
            int32_t x_25559 = lshr32(res_25558, 16);
            int32_t x_25560 = res_25558 ^ x_25559;
            int32_t x_25561 = mul32(73244475, x_25560);
            int32_t x_25562 = lshr32(x_25561, 16);
            int32_t x_25563 = x_25561 ^ x_25562;
            int32_t x_25564 = mul32(73244475, x_25563);
            int32_t x_25565 = lshr32(x_25564, 16);
            int32_t x_25566 = x_25564 ^ x_25565;
            int32_t unsign_arg_25567 = unsign_arg_25555 ^ x_25566;
            int32_t unsign_arg_25568 = mul32(48271, unsign_arg_25567);
            int32_t unsign_arg_25569 = umod32(unsign_arg_25568, 2147483647);
            int32_t unsign_arg_25570 = mul32(48271, unsign_arg_25569);
            int32_t unsign_arg_25571 = umod32(unsign_arg_25570, 2147483647);
            float res_25572 = uitofp_i32_f32(unsign_arg_25569);
            float res_25573 = res_25572 / 2.1474836e9F;
            float res_25574 = uitofp_i32_f32(unsign_arg_25571);
            float res_25575 = res_25574 / 2.1474836e9F;
            float res_25576;
            
            res_25576 = futrts_log32(res_25573);
            
            float res_25577 = -2.0F * res_25576;
            float res_25578;
            
            res_25578 = futrts_sqrt32(res_25577);
            
            float res_25579 = 6.2831855F * res_25575;
            float res_25580;
            
            res_25580 = futrts_cos32(res_25579);
            
            float res_25581 = res_25578 * res_25580;
            
            ((float *) mem_26255)[i_25899] = res_25581;
        }
        memmove(mem_26243 + i_25903 * steps_25439 * 4, mem_26269 + 0,
                steps_25439 * (int32_t) sizeof(float));
        for (int64_t i_25584 = 0; i_25584 < upper_bound_25540; i_25584++) {
            bool y_25586 = slt64(i_25584, steps_25439);
            bool index_certs_25587;
            
            if (!y_25586) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25584,
                              "] out of bounds for array of shape [",
                              steps_25439, "].",
                              "-> #0  cva.fut:71:97-104\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:173:3-120\n   #4  cva.fut:172:1-173:128\n");
                return 1;
            }
            
            float shortstep_arg_25588 = ((float *) mem_26255)[i_25584];
            float shortstep_arg_25589 = ((float *) mem_26243)[i_25903 *
                                                              steps_25439 +
                                                              i_25584];
            float y_25590 = 5.0e-2F - shortstep_arg_25589;
            float x_25591 = 1.0e-2F * y_25590;
            float x_25592 = dt_25452 * x_25591;
            float x_25593 = res_25541 * shortstep_arg_25588;
            float y_25594 = 1.0e-3F * x_25593;
            float delta_r_25595 = x_25592 + y_25594;
            float res_25596 = shortstep_arg_25589 + delta_r_25595;
            int64_t i_25597 = add64(1, i_25584);
            bool x_25598 = sle64(0, i_25597);
            bool y_25599 = slt64(i_25597, steps_25439);
            bool bounds_check_25600 = x_25598 && y_25599;
            bool index_certs_25601;
            
            if (!bounds_check_25600) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25597,
                              "] out of bounds for array of shape [",
                              steps_25439, "].",
                              "-> #0  cva.fut:71:58-105\n   #1  cva.fut:123:32-62\n   #2  cva.fut:123:22-69\n   #3  cva.fut:173:3-120\n   #4  cva.fut:172:1-173:128\n");
                return 1;
            }
            ((float *) mem_26243)[i_25903 * steps_25439 + i_25597] = res_25596;
        }
    }
    
    int64_t flat_dim_25605 = 9 * paths_25438;
    float res_25609 = sitofp_i64_f32(paths_25438);
    int64_t binop_x_26297 = 9 * paths_25438;
    int64_t bytes_26296 = 4 * binop_x_26297;
    
    if (mem_26298_cached_sizze_26517 < (size_t) bytes_26296) {
        mem_26298 = realloc(mem_26298, bytes_26296);
        mem_26298_cached_sizze_26517 = bytes_26296;
    }
    if (mem_26313_cached_sizze_26518 < (size_t) 36) {
        mem_26313 = realloc(mem_26313, 36);
        mem_26313_cached_sizze_26518 = 36;
    }
    
    int64_t bytes_26322 = 8 * flat_dim_25605;
    
    if (mem_26323_cached_sizze_26519 < (size_t) bytes_26322) {
        mem_26323 = realloc(mem_26323, bytes_26322);
        mem_26323_cached_sizze_26519 = bytes_26322;
    }
    if (mem_26325_cached_sizze_26520 < (size_t) bytes_26322) {
        mem_26325 = realloc(mem_26325, bytes_26322);
        mem_26325_cached_sizze_26520 = bytes_26322;
    }
    
    float res_25611;
    float redout_26005 = 0.0F;
    
    for (int64_t i_26006 = 0; i_26006 < steps_25439; i_26006++) {
        int64_t index_primexp_26145 = add64(1, i_26006);
        float res_25617 = sitofp_i64_f32(index_primexp_26145);
        float res_25618 = res_25617 / sims_per_year_25530;
        
        for (int64_t i_25907 = 0; i_25907 < paths_25438; i_25907++) {
            float x_25620 = ((float *) mem_26243)[i_25907 * steps_25439 +
                                                  i_26006];
            
            for (int64_t i_26464 = 0; i_26464 < 9; i_26464++) {
                ((float *) mem_26313)[i_26464] = x_25620;
            }
            memmove(mem_26298 + i_25907 * 9 * 4, mem_26313 + 0, 9 *
                    (int32_t) sizeof(float));
        }
        
        int64_t discard_25917;
        int64_t scanacc_25911 = 0;
        
        for (int64_t i_25914 = 0; i_25914 < flat_dim_25605; i_25914++) {
            int64_t binop_x_26025 = squot64(i_25914, 9);
            int64_t binop_y_26026 = 9 * binop_x_26025;
            int64_t new_index_26027 = i_25914 - binop_y_26026;
            int64_t x_25627 = ((int64_t *) mem_26168)[new_index_26027];
            float x_25628 = ((float *) mem_26170)[new_index_26027];
            float x_25629 = res_25618 / x_25628;
            float ceil_arg_25630 = x_25629 - 1.0F;
            float res_25631;
            
            res_25631 = futrts_ceil32(ceil_arg_25630);
            
            int64_t res_25632 = fptosi_f32_i64(res_25631);
            int64_t max_arg_25633 = sub64(x_25627, res_25632);
            int64_t res_25634 = smax64(0, max_arg_25633);
            bool cond_25635 = res_25634 == 0;
            int64_t res_25636;
            
            if (cond_25635) {
                res_25636 = 1;
            } else {
                res_25636 = res_25634;
            }
            
            int64_t res_25626 = add64(res_25636, scanacc_25911);
            
            ((int64_t *) mem_26323)[i_25914] = res_25626;
            ((int64_t *) mem_26325)[i_25914] = res_25636;
            
            int64_t scanacc_tmp_26465 = res_25626;
            
            scanacc_25911 = scanacc_tmp_26465;
        }
        discard_25917 = scanacc_25911;
        
        int64_t res_25638;
        int64_t redout_25918 = 0;
        
        for (int64_t i_25919 = 0; i_25919 < flat_dim_25605; i_25919++) {
            int64_t x_25642 = ((int64_t *) mem_26325)[i_25919];
            int64_t res_25641 = add64(x_25642, redout_25918);
            int64_t redout_tmp_26468 = res_25641;
            
            redout_25918 = redout_tmp_26468;
        }
        res_25638 = redout_25918;
        
        bool bounds_invalid_upwards_25643 = slt64(res_25638, 0);
        bool valid_25644 = !bounds_invalid_upwards_25643;
        bool range_valid_c_25645;
        
        if (!valid_25644) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", res_25638,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:48:30-60\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:87:14-32\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:132:34-93\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:126:18-137:50\n   #9  cva.fut:173:3-120\n   #10 cva.fut:172:1-173:128\n");
            return 1;
        }
        
        int64_t bytes_26350 = 8 * res_25638;
        
        if (mem_26351_cached_sizze_26521 < (size_t) bytes_26350) {
            mem_26351 = realloc(mem_26351, bytes_26350);
            mem_26351_cached_sizze_26521 = bytes_26350;
        }
        for (int64_t i_26469 = 0; i_26469 < res_25638; i_26469++) {
            ((int64_t *) mem_26351)[i_26469] = 0;
        }
        for (int64_t iter_25920 = 0; iter_25920 < flat_dim_25605;
             iter_25920++) {
            int64_t i_p_o_26032 = add64(-1, iter_25920);
            int64_t rot_i_26033 = smod64(i_p_o_26032, flat_dim_25605);
            int64_t pixel_25923 = ((int64_t *) mem_26323)[rot_i_26033];
            bool cond_25653 = iter_25920 == 0;
            int64_t res_25654;
            
            if (cond_25653) {
                res_25654 = 0;
            } else {
                res_25654 = pixel_25923;
            }
            
            bool less_than_zzero_25924 = slt64(res_25654, 0);
            bool greater_than_sizze_25925 = sle64(res_25638, res_25654);
            bool outside_bounds_dim_25926 = less_than_zzero_25924 ||
                 greater_than_sizze_25925;
            
            if (!outside_bounds_dim_25926) {
                int64_t read_hist_25928 = ((int64_t *) mem_26351)[res_25654];
                int64_t res_25650 = smax64(iter_25920, read_hist_25928);
                
                ((int64_t *) mem_26351)[res_25654] = res_25650;
            }
        }
        if (mem_26365_cached_sizze_26522 < (size_t) bytes_26350) {
            mem_26365 = realloc(mem_26365, bytes_26350);
            mem_26365_cached_sizze_26522 = bytes_26350;
        }
        
        int64_t discard_25941;
        int64_t scanacc_25934 = 0;
        
        for (int64_t i_25937 = 0; i_25937 < res_25638; i_25937++) {
            int64_t x_25664 = ((int64_t *) mem_26351)[i_25937];
            bool res_25665 = slt64(0, x_25664);
            int64_t res_25662;
            
            if (res_25665) {
                res_25662 = x_25664;
            } else {
                int64_t res_25663 = add64(x_25664, scanacc_25934);
                
                res_25662 = res_25663;
            }
            ((int64_t *) mem_26365)[i_25937] = res_25662;
            
            int64_t scanacc_tmp_26471 = res_25662;
            
            scanacc_25934 = scanacc_tmp_26471;
        }
        discard_25941 = scanacc_25934;
        
        int64_t bytes_26378 = 4 * res_25638;
        
        if (mem_26379_cached_sizze_26523 < (size_t) bytes_26378) {
            mem_26379 = realloc(mem_26379, bytes_26378);
            mem_26379_cached_sizze_26523 = bytes_26378;
        }
        if (mem_26381_cached_sizze_26524 < (size_t) res_25638) {
            mem_26381 = realloc(mem_26381, res_25638);
            mem_26381_cached_sizze_26524 = res_25638;
        }
        
        int64_t inpacc_25674;
        float inpacc_25676;
        int64_t inpacc_25682;
        float inpacc_25684;
        
        inpacc_25682 = 0;
        inpacc_25684 = 0.0F;
        for (int64_t i_25980 = 0; i_25980 < res_25638; i_25980++) {
            int64_t x_26047 = ((int64_t *) mem_26365)[i_25980];
            int64_t i_p_o_26049 = add64(-1, i_25980);
            int64_t rot_i_26050 = smod64(i_p_o_26049, res_25638);
            int64_t x_26051 = ((int64_t *) mem_26365)[rot_i_26050];
            bool res_26052 = x_26047 == x_26051;
            bool res_26053 = !res_26052;
            int64_t res_25718;
            
            if (res_26053) {
                res_25718 = 1;
            } else {
                int64_t res_25719 = add64(1, inpacc_25682);
                
                res_25718 = res_25719;
            }
            
            int64_t res_25733;
            
            if (res_26053) {
                res_25733 = 1;
            } else {
                int64_t res_25734 = add64(1, inpacc_25682);
                
                res_25733 = res_25734;
            }
            
            int64_t res_25735 = sub64(res_25733, 1);
            bool x_26068 = sle64(0, x_26047);
            bool y_26069 = slt64(x_26047, flat_dim_25605);
            bool bounds_check_26070 = x_26068 && y_26069;
            bool index_certs_26071;
            
            if (!bounds_check_26070) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", x_26047,
                              "] out of bounds for array of shape [",
                              flat_dim_25605, "].",
                              "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:90:30-35\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:90:12-49\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:132:34-93\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:126:18-137:50\n   #9  cva.fut:173:3-120\n   #10 cva.fut:172:1-173:128\n");
                return 1;
            }
            
            int64_t new_index_26072 = squot64(x_26047, 9);
            int64_t binop_y_26073 = 9 * new_index_26072;
            int64_t new_index_26074 = x_26047 - binop_y_26073;
            float lifted_0_get_arg_26075 =
                  ((float *) mem_26298)[new_index_26072 * 9 + new_index_26074];
            float lifted_0_get_arg_26076 =
                  ((float *) mem_26164)[new_index_26074];
            float lifted_0_get_arg_26077 =
                  ((float *) mem_26166)[new_index_26074];
            int64_t lifted_0_get_arg_26078 =
                    ((int64_t *) mem_26168)[new_index_26074];
            float lifted_0_get_arg_26079 =
                  ((float *) mem_26170)[new_index_26074];
            float x_26080 = res_25618 / lifted_0_get_arg_26079;
            float ceil_arg_26081 = x_26080 - 1.0F;
            float res_26082;
            
            res_26082 = futrts_ceil32(ceil_arg_26081);
            
            int64_t res_26083 = fptosi_f32_i64(res_26082);
            int64_t max_arg_26084 = sub64(lifted_0_get_arg_26078, res_26083);
            int64_t res_26085 = smax64(0, max_arg_26084);
            bool cond_26086 = res_26085 == 0;
            float res_26087;
            
            if (cond_26086) {
                res_26087 = 0.0F;
            } else {
                float res_26088;
                
                res_26088 = futrts_ceil32(x_26080);
                
                float start_26089 = lifted_0_get_arg_26079 * res_26088;
                float res_26090;
                
                res_26090 = futrts_ceil32(ceil_arg_26081);
                
                int64_t res_26091 = fptosi_f32_i64(res_26090);
                int64_t max_arg_26092 = sub64(lifted_0_get_arg_26078,
                                              res_26091);
                int64_t res_26093 = smax64(0, max_arg_26092);
                int64_t sizze_26094 = sub64(res_26093, 1);
                bool cond_26095 = res_25735 == 0;
                float res_26096;
                
                if (cond_26095) {
                    res_26096 = 1.0F;
                } else {
                    res_26096 = 0.0F;
                }
                
                bool cond_26097 = slt64(0, res_25735);
                float res_26098;
                
                if (cond_26097) {
                    float y_26099 = lifted_0_get_arg_26076 *
                          lifted_0_get_arg_26079;
                    float res_26100 = res_26096 - y_26099;
                    
                    res_26098 = res_26100;
                } else {
                    res_26098 = res_26096;
                }
                
                bool cond_26101 = res_25735 == sizze_26094;
                float res_26102;
                
                if (cond_26101) {
                    float res_26103 = res_26098 - 1.0F;
                    
                    res_26102 = res_26103;
                } else {
                    res_26102 = res_26098;
                }
                
                float res_26104 = lifted_0_get_arg_26077 * res_26102;
                float res_26105 = sitofp_i64_f32(res_25735);
                float y_26106 = lifted_0_get_arg_26079 * res_26105;
                float bondprice_arg_26107 = start_26089 + y_26106;
                float y_26108 = bondprice_arg_26107 - res_25618;
                float negate_arg_26109 = 1.0e-2F * y_26108;
                float exp_arg_26110 = 0.0F - negate_arg_26109;
                float res_26111 = fpow32(2.7182817F, exp_arg_26110);
                float x_26112 = 1.0F - res_26111;
                float B_26113 = x_26112 / 1.0e-2F;
                float x_26114 = B_26113 - bondprice_arg_26107;
                float x_26115 = res_25618 + x_26114;
                float x_26116 = 4.4999997e-6F * x_26115;
                float A1_26117 = x_26116 / 1.0e-4F;
                float y_26118 = fpow32(B_26113, 2.0F);
                float x_26119 = 1.0000001e-6F * y_26118;
                float A2_26120 = x_26119 / 4.0e-2F;
                float exp_arg_26121 = A1_26117 - A2_26120;
                float res_26122 = fpow32(2.7182817F, exp_arg_26121);
                float negate_arg_26123 = lifted_0_get_arg_26075 * B_26113;
                float exp_arg_26124 = 0.0F - negate_arg_26123;
                float res_26125 = fpow32(2.7182817F, exp_arg_26124);
                float res_26126 = res_26122 * res_26125;
                float res_26127 = res_26104 * res_26126;
                
                res_26087 = res_26127;
            }
            
            float res_25805;
            
            if (res_26053) {
                res_25805 = res_26087;
            } else {
                float res_25806 = inpacc_25684 + res_26087;
                
                res_25805 = res_25806;
            }
            
            float res_25808;
            
            if (res_26053) {
                res_25808 = res_26087;
            } else {
                float res_25809 = inpacc_25684 + res_26087;
                
                res_25808 = res_25809;
            }
            ((float *) mem_26379)[i_25980] = res_25805;
            ((bool *) mem_26381)[i_25980] = res_26053;
            
            int64_t inpacc_tmp_26473 = res_25718;
            float inpacc_tmp_26474 = res_25808;
            
            inpacc_25682 = inpacc_tmp_26473;
            inpacc_25684 = inpacc_tmp_26474;
        }
        inpacc_25674 = inpacc_25682;
        inpacc_25676 = inpacc_25684;
        if (mem_26407_cached_sizze_26525 < (size_t) bytes_26350) {
            mem_26407 = realloc(mem_26407, bytes_26350);
            mem_26407_cached_sizze_26525 = bytes_26350;
        }
        
        int64_t discard_25989;
        int64_t scanacc_25985 = 0;
        
        for (int64_t i_25987 = 0; i_25987 < res_25638; i_25987++) {
            int64_t i_p_o_26138 = add64(1, i_25987);
            int64_t rot_i_26139 = smod64(i_p_o_26138, res_25638);
            bool x_25822 = ((bool *) mem_26381)[rot_i_26139];
            int64_t res_25823 = btoi_bool_i64(x_25822);
            int64_t res_25821 = add64(res_25823, scanacc_25985);
            
            ((int64_t *) mem_26407)[i_25987] = res_25821;
            
            int64_t scanacc_tmp_26477 = res_25821;
            
            scanacc_25985 = scanacc_tmp_26477;
        }
        discard_25989 = scanacc_25985;
        
        bool cond_25824 = slt64(0, res_25638);
        int64_t num_segments_25825;
        
        if (cond_25824) {
            int64_t i_25826 = sub64(res_25638, 1);
            bool x_25827 = sle64(0, i_25826);
            bool y_25828 = slt64(i_25826, res_25638);
            bool bounds_check_25829 = x_25827 && y_25828;
            bool index_certs_25830;
            
            if (!bounds_check_25829) {
                ctx->error =
                    msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                              "Index [", i_25826,
                              "] out of bounds for array of shape [", res_25638,
                              "].",
                              "-> #0  /prelude/array.fut:18:29-34\n   #1  lib/github.com/diku-dk/segmented/segmented.fut:29:36-59\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #4  cva.fut:132:34-93\n   #5  /prelude/soacs.fut:56:19-23\n   #6  /prelude/soacs.fut:56:3-37\n   #7  cva.fut:126:18-137:50\n   #8  cva.fut:173:3-120\n   #9  cva.fut:172:1-173:128\n");
                return 1;
            }
            
            int64_t res_25831 = ((int64_t *) mem_26407)[i_25826];
            
            num_segments_25825 = res_25831;
        } else {
            num_segments_25825 = 0;
        }
        
        bool bounds_invalid_upwards_25832 = slt64(num_segments_25825, 0);
        bool valid_25833 = !bounds_invalid_upwards_25832;
        bool range_valid_c_25834;
        
        if (!valid_25833) {
            ctx->error =
                msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                          "Range ", 0, "..", 1, "..<", num_segments_25825,
                          " is invalid.",
                          "-> #0  /prelude/array.fut:60:3-10\n   #1  /prelude/array.fut:70:18-23\n   #2  lib/github.com/diku-dk/segmented/segmented.fut:33:17-41\n   #3  lib/github.com/diku-dk/segmented/segmented.fut:91:6-36\n   #4  lib/github.com/diku-dk/segmented/segmented.fut:103:6-37\n   #5  cva.fut:132:34-93\n   #6  /prelude/soacs.fut:56:19-23\n   #7  /prelude/soacs.fut:56:3-37\n   #8  cva.fut:126:18-137:50\n   #9  cva.fut:173:3-120\n   #10 cva.fut:172:1-173:128\n");
            return 1;
        }
        
        int64_t bytes_26420 = 4 * num_segments_25825;
        
        if (mem_26421_cached_sizze_26526 < (size_t) bytes_26420) {
            mem_26421 = realloc(mem_26421, bytes_26420);
            mem_26421_cached_sizze_26526 = bytes_26420;
        }
        for (int64_t i_26479 = 0; i_26479 < num_segments_25825; i_26479++) {
            ((float *) mem_26421)[i_26479] = 0.0F;
        }
        for (int64_t write_iter_25990 = 0; write_iter_25990 < res_25638;
             write_iter_25990++) {
            int64_t write_iv_25992 = ((int64_t *) mem_26407)[write_iter_25990];
            int64_t i_p_o_26141 = add64(1, write_iter_25990);
            int64_t rot_i_26142 = smod64(i_p_o_26141, res_25638);
            bool write_iv_25993 = ((bool *) mem_26381)[rot_i_26142];
            int64_t res_25840;
            
            if (write_iv_25993) {
                int64_t res_25841 = sub64(write_iv_25992, 1);
                
                res_25840 = res_25841;
            } else {
                res_25840 = -1;
            }
            
            bool less_than_zzero_25995 = slt64(res_25840, 0);
            bool greater_than_sizze_25996 = sle64(num_segments_25825,
                                                  res_25840);
            bool outside_bounds_dim_25997 = less_than_zzero_25995 ||
                 greater_than_sizze_25996;
            
            if (!outside_bounds_dim_25997) {
                memmove(mem_26421 + res_25840 * 4, mem_26379 +
                        write_iter_25990 * 4, (int32_t) sizeof(float));
            }
        }
        
        bool dim_match_25842 = flat_dim_25605 == num_segments_25825;
        bool empty_or_match_cert_25843;
        
        if (!dim_match_25842) {
            ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                                   "Value of (core language) shape (",
                                   num_segments_25825,
                                   ") cannot match shape of type `[",
                                   flat_dim_25605, "]b`.",
                                   "-> #0  lib/github.com/diku-dk/segmented/segmented.fut:103:6-45\n   #1  cva.fut:132:34-93\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  cva.fut:126:18-137:50\n   #5  cva.fut:173:3-120\n   #6  cva.fut:172:1-173:128\n");
            return 1;
        }
        
        float res_25845;
        float redout_26003 = 0.0F;
        
        for (int64_t i_26004 = 0; i_26004 < paths_25438; i_26004++) {
            int64_t binop_x_26146 = 9 * i_26004;
            float res_25850;
            float redout_26001 = 0.0F;
            
            for (int32_t i_26143 = 0; i_26143 < 9; i_26143++) {
                int64_t i_26002 = sext_i32_i64(i_26143);
                int64_t new_index_26147 = i_26002 + binop_x_26146;
                float x_25854 = ((float *) mem_26421)[new_index_26147];
                float res_25853 = x_25854 + redout_26001;
                float redout_tmp_26482 = res_25853;
                
                redout_26001 = redout_tmp_26482;
            }
            res_25850 = redout_26001;
            
            float res_25855 = fmax32(0.0F, res_25850);
            float res_25848 = res_25855 + redout_26003;
            float redout_tmp_26481 = res_25848;
            
            redout_26003 = redout_tmp_26481;
        }
        res_25845 = redout_26003;
        
        float res_25856 = res_25845 / res_25609;
        float negate_arg_25857 = 1.0e-2F * res_25618;
        float exp_arg_25858 = 0.0F - negate_arg_25857;
        float res_25859 = fpow32(2.7182817F, exp_arg_25858);
        float x_25860 = 1.0F - res_25859;
        float B_25861 = x_25860 / 1.0e-2F;
        float x_25862 = B_25861 - res_25618;
        float x_25863 = 4.4999997e-6F * x_25862;
        float A1_25864 = x_25863 / 1.0e-4F;
        float y_25865 = fpow32(B_25861, 2.0F);
        float x_25866 = 1.0000001e-6F * y_25865;
        float A2_25867 = x_25866 / 4.0e-2F;
        float exp_arg_25868 = A1_25864 - A2_25867;
        float res_25869 = fpow32(2.7182817F, exp_arg_25868);
        float negate_arg_25870 = 5.0e-2F * B_25861;
        float exp_arg_25871 = 0.0F - negate_arg_25870;
        float res_25872 = fpow32(2.7182817F, exp_arg_25871);
        float res_25873 = res_25869 * res_25872;
        float res_25874 = res_25856 * res_25873;
        float res_25614 = res_25874 + redout_26005;
        float redout_tmp_26462 = res_25614;
        
        redout_26005 = redout_tmp_26462;
    }
    res_25611 = redout_26005;
    
    float CVA_25875 = 6.0e-3F * res_25611;
    
    scalar_out_26447 = CVA_25875;
    *out_scalar_out_26505 = scalar_out_26447;
    
  cleanup:
    { }
    free(mem_26158);
    free(mem_26160);
    free(mem_26162);
    free(mem_26164);
    free(mem_26166);
    free(mem_26168);
    free(mem_26170);
    free(mem_26208);
    free(mem_26243);
    free(mem_26255);
    free(mem_26269);
    free(mem_26298);
    free(mem_26313);
    free(mem_26323);
    free(mem_26325);
    free(mem_26351);
    free(mem_26365);
    free(mem_26379);
    free(mem_26381);
    free(mem_26407);
    free(mem_26421);
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
    struct memblock swap_term_mem_26157;
    
    swap_term_mem_26157.references = NULL;
    
    struct memblock payments_mem_26158;
    
    payments_mem_26158.references = NULL;
    
    struct memblock notional_mem_26159;
    
    notional_mem_26159.references = NULL;
    
    int64_t n_24721;
    int64_t n_24722;
    int64_t n_24723;
    int64_t paths_24724;
    int64_t steps_24725;
    float a_24729;
    float b_24730;
    float sigma_24731;
    float r0_24732;
    float scalar_out_26447;
    struct memblock out_mem_26448;
    
    out_mem_26448.references = NULL;
    
    int64_t out_arrsizze_26449;
    
    lock_lock(&ctx->lock);
    paths_24724 = in0;
    steps_24725 = in1;
    swap_term_mem_26157 = in2->mem;
    n_24721 = in2->shape[0];
    payments_mem_26158 = in3->mem;
    n_24722 = in3->shape[0];
    notional_mem_26159 = in4->mem;
    n_24723 = in4->shape[0];
    a_24729 = in5;
    b_24730 = in6;
    sigma_24731 = in7;
    r0_24732 = in8;
    
    int ret = futrts_main(ctx, &scalar_out_26447, &out_mem_26448,
                          &out_arrsizze_26449, swap_term_mem_26157,
                          payments_mem_26158, notional_mem_26159, n_24721,
                          n_24722, n_24723, paths_24724, steps_24725, a_24729,
                          b_24730, sigma_24731, r0_24732);
    
    if (ret == 0) {
        *out0 = scalar_out_26447;
        assert((*out1 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out1)->mem = out_mem_26448;
        (*out1)->shape[0] = out_arrsizze_26449;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test(struct futhark_context *ctx, float *out0, const
                       int64_t in0, const int64_t in1)
{
    int64_t paths_25438;
    int64_t steps_25439;
    float scalar_out_26447;
    
    lock_lock(&ctx->lock);
    paths_25438 = in0;
    steps_25439 = in1;
    
    int ret = futrts_test(ctx, &scalar_out_26447, paths_25438, steps_25439);
    
    if (ret == 0) {
        *out0 = scalar_out_26447;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
