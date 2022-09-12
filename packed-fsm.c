#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

#if !defined NDEBUG
static void print_input(const uint8_t ptr[16])
{
  printf("input: ");
  for (size_t i = 0; i < 16; i++) {
    printf(" %c ", ptr[i]);
  }
  printf("\n");
}

static void print_mm_epi8(const __m128i *mm)
{
  printf("mm:    ");
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

static void print_packed_mm_epi8(const __m128i *mm)
{
  printf("mm/2:  ");
  for (size_t i=0; i < 16; i += 2) {
    const uint8_t b = ((uint8_t *)mm)[i];
    printf("%s%x%x", i ? " " : "", (int)b&0x0f, ((int)b&0xf0)>>4);
  }
  printf("\n");
}

// Classify and pack characters in the set so we're left with transitions.
//
// Based on the simdjson paper and the "Special case 1 - small sets".
// https://arxiv.org/abs/1902.08318
// http://0x80.pl/articles/simd-byte-lookup.html
//
//  <*********>  :  0x00  :  0x00 -- contiguous
//  " "  : 0x20  :  0x01  :  0x01 |- space
//  "\t" : 0x09  :  0x02  :  0x01 |
//  "\r" : 0x0d  :  0x02  :  0x01 |
//  "\n" : 0x0a  :  0x04  :  0x02 -- newline (ends record and comment)
//  "\"" : 0x22  :  0x08  :  0x03 -- starts and ends quoted
//  ";"  : 0x3b  :  0x10  :  0x04 -- starts comment
//  "\\" : 0x5c  :  0x20  :  0x05 -- next character is escaped (except in comment)
//  "("  : 0x28  :  0x40  :  0x06 -- starts grouped (not a scanner state)
//  ")"  : 0x29  :  0x80  :  0x06 -- end grouped (not a scanner state)
static inline __m128i classify(const uint8_t ptr[16])
{
  const __m128i input = _mm_loadu_si128((const __m128i *)ptr);

  const __m128i input_hi = _mm_and_si128(_mm_srli_epi16(input, 4), _mm_set1_epi8(0x0f));

  // round 1
  static const uint8_t r1_lo[16] = { 0x01,0x00,0x08,0x00,0x00,0x00,0x00,0x00,
                                     0x40,0x82,0x04,0x10,0x20,0x02,0x00,0x00 };
  static const uint8_t r1_hi[16] = { 0x06,0x00,0xc9,0x10,0x00,0x20,0x00,0x00,
                                     0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };

  const __m128i x1_lo = _mm_shuffle_epi8(
    _mm_load_si128((const __m128i *)r1_lo), input);
  const __m128i x1_hi = _mm_shuffle_epi8(
    _mm_load_si128((const __m128i *)r1_hi), input_hi);
  const __m128i x1 = _mm_and_si128(x1_lo, x1_hi);
  print_mm_epi8(&x1);

  // round 2
  static const uint8_t r2_lo[16] = { 0,1,1,0,2,0,0,0, 3,0,0,0,0,0,0,0 };
  static const uint8_t r2_hi[16] = { 0,4,5,0,6,0,0,0, 6,0,0,0,0,0,0,0 };

  const __m128i x2_lo = _mm_shuffle_epi8(
    _mm_load_si128((const __m128i *)r2_lo), x1);
  const __m128i x2_hi = _mm_shuffle_epi8(
    _mm_load_si128((const __m128i *)r2_hi), _mm_srli_epi16(x1, 4));
  const __m128i x2 = _mm_or_si128(x2_lo, x2_hi);
  print_mm_epi8(&x2);

  // pack
  return _mm_or_si128(_mm_srli_epi16(x2, 4), x2);
}

#include "transitions.h"

#if !defined NDEBUG
static void print_transition(uint8_t index, const transition_t t)
{
  printf("%0.2x  { .next = %u, .mask = %u }\n", index, t.next, t.mask);
}
#else
#define print_transition(t) /* empty */
#endif

static inline uint16_t mask(const __m128i *restrict packed, uint32_t *state)
{
  uint16_t m;

  const uint8_t t0 = *state;
  const transition_t t1 = transitions[t0][ ((const uint8_t *)packed)[0]  ];
  print_transition(((const uint8_t *)packed)[0], t1);
  m  = t1.mask << 14;
  const transition_t t2 = transitions[t1.next][ ((const uint8_t *)packed)[2]  ];
  print_transition(((const uint8_t *)packed)[2], t2);
  m |= t2.mask << 12;
  const transition_t t3 = transitions[t2.next][ ((const uint8_t *)packed)[4]  ];
  print_transition(((const uint8_t *)packed)[4], t3);
  m |= t3.mask << 10;
  const transition_t t4 = transitions[t3.next][ ((const uint8_t *)packed)[6]  ];
  print_transition(((const uint8_t *)packed)[6], t4);
  m |= t4.mask <<  8;
  const transition_t t5 = transitions[t4.next][ ((const uint8_t *)packed)[8]  ];
  print_transition(((const uint8_t *)packed)[8], t5);
  m |= t5.mask <<  6;
  const transition_t t6 = transitions[t5.next][ ((const uint8_t *)packed)[10] ];
  print_transition(((const uint8_t *)packed)[10], t6);
  m |= t6.mask <<  4;
  const transition_t t7 = transitions[t6.next][ ((const uint8_t *)packed)[12] ];
  print_transition(((const uint8_t *)packed)[12], t7);
  m |= t7.mask <<  2;
  const transition_t t8 = transitions[t7.next][ ((const uint8_t *)packed)[14] ];
  print_transition(((const uint8_t *)packed)[14], t8);
  m |= t8.mask;

  *state = t8.next;
  return m;
}

void print_mask(uint16_t mask) {
  printf("[ ");
  for(int i = 0, n = (sizeof(mask)*8)-1; i <= n; i++){
    char c = (mask &(1LL<<i))? '1' : '0';
    putchar(c);
  }
  printf(" ]\n");
}

#include "transitions.h"

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  const uint8_t input[16] =
    { 'a',' ','2','"',  '(',' ','d',')',  ' ',';','f','g',  'h','i','j','k' };

  print_input(input);
  __m128i packed = classify(input);
  print_packed_mm_epi8(&packed);
  uint32_t state = 0;
  uint16_t bitmask = mask(&packed, &state);
  printf("input:      %.*s\n", sizeof(input), input);
  printf("result:   ");
  print_mask(bitmask);

  return 0;
}
