#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> // assume x86_64 for now

void print_mask(uint16_t mask) {
  printf("[ ");
  for(int i = 0, n = (sizeof(mask)*8)-1; i <= n; i++){
    char c = (mask &(1LL<<i))? '1' : '0';
    putchar(c);
  }
  printf(" ]\n");
}


#if !defined NDEBUG
static void print_input(const uint8_t ptr[16])
{
  printf("input: ");
  for (size_t i = 0; i < 16; i++) {
    printf(" %c ", ptr[i]);
  }
  printf("\n");
}

static void print_mm_epi8(int x, const __m128i *mm)
{
  printf("%d mm:  ", x);
  for (size_t i=0; i < 16; i++) {
    const uint8_t b = ((uint8_t *)mm)[i];
    printf("%s%0.2x", i ? " " : "", (int)b);
  }
  printf("\n");
}
#else
#define print_input(x)
#define print_mm_epi8(x)
#endif

// Based on simdjson (1)
// 1: https://arxiv.org/abs/1902.08318
static inline uint16_t find_escaped(const uint8_t ptr[16])
{
  const uint16_t e = 0x5555U;

  // Find backslashes
  __m128i backslash = _mm_cmpeq_epi8(_mm_set1_epi8('\\'), _mm_loadu_si128((__m128i*)ptr));
  uint16_t b = _mm_movemask_epi8(backslash);

  // Identify 'starts' - backslash characters not preceded by backslashes.
  uint16_t s = b & ~(b << 1);
  // Detect end of a odd-length sequence of backslashes starting on an even
  // offset.
  // Detail: ES gets all 'starts' that begin on even offsets
  uint16_t es = s & e;
  // Add B to ES, yielding carries on backslash sequences with event starts.
  uint16_t ec = b + es;
  // Filter out the backslashes from the previous addition, getting carries
  // only.
  uint16_t ece = ec & ~b;
  // Select only the end of sequences ending on an odd offset.
  uint16_t od1 = ece & ~e;

  // Detect end of a odd-length sequence of backslashes starting on an odd
  // offset details are as per the above sequence.
  uint16_t os = s & ~e;
  uint16_t oc = b + os;
  uint16_t oce = oc & ~b;
  uint16_t od2 = oce & e;

  // Merge results, yielding ends of all odd-length sequence of backslashes.
  uint16_t od = od1 | od2;

  return od;
}

// Based on Stack Overflow question #67201469 (1)
// 1: https://stackoverflow.com/questions/67201469
static inline __m128i mask_to_m128i(const uint16_t bitmask)
{
  // Broadcast mask to all elements
  const __m128i bitmasks = _mm_shuffle_epi8(
    _mm_cvtsi32_si128(bitmask), _mm_setr_epi32(0, 0, 0x01010101, 0x01010101));

  // Select correct bit in each element
  const __m128i bitselect = _mm_setr_epi8(
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);

  return _mm_cmpeq_epi8(
    _mm_setzero_si128(), _mm_and_si128(bitmasks, bitselect));
}

// Classify and pack characters so the transitions are left.
//
// Based on the simdjson (1) paper and the "Special case 1 - small sets" (2).
// 1: https://arxiv.org/abs/1902.08318
// 2: http://0x80.pl/articles/simd-byte-lookup.html
//
// <**********> : 0x00 -- contiguous
// "\n" : 0x0a  : 0x01 -- newline (ends record and comment)
// "\"" : 0x22  : 0x02 -- starts and ends quoted
// ";"  : 0x3b  : 0x03 -- starts a comment
static inline void classify(const uint8_t src[16])
{
  const __m128i lo_src = _mm_loadu_si128((const __m128i *)src);
  const __m128i hi_src = _mm_and_si128(
    _mm_srli_epi16(lo_src, 4), _mm_set1_epi8(0x0f));

  static const uint8_t hi_mask[16] = {
    // 0     1     2     3     4     5     6     7
       0x01, 0x00, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00,
    // 8     9     a     b     c     d     e     f
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
  };

  static const uint8_t lo_mask[16] = {
    // 0     1     2     3     4     5     6     7
       0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
    // 8     9     a     b     c     d     e     f
       0x00, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x00
  };

  const __m128i hi_xlat = _mm_shuffle_epi8(
    _mm_load_si128((const __m128i *)hi_mask), hi_src);
  const __m128i lo_xlat = _mm_shuffle_epi8(
    _mm_load_si128((const __m128i *)lo_mask), lo_src);
  __m128i xlat = _mm_and_si128(hi_xlat, lo_xlat);
  print_mm_epi8(1, &xlat);

  // Apply simdjson backslash trick to discard escaped strings/comments.
  // Done at this stage to allow packing transitions in two bits.
  const __m128i esc = mask_to_m128i(find_escaped(src));
  print_mm_epi8(2, &esc);

  xlat = _mm_and_si128(xlat, esc);
  print_mm_epi8(3, &xlat);
}

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  const uint8_t input[16] =
    { '\\','a','\\','"',  '(','d','\\',')',  ' ',';','f','g',  'h','i','k',';' };

  print_input(input);
  classify(input);

  return 0;
}
