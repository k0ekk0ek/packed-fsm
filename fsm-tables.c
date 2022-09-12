#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define INITIAL (0x00)
#define CONTIGUOUS (0x01)
#define QUOTED (0x02)
#define WHITESPACE (0x03) // both state and character class
#define COMMENT (0x04)
#define ESCAPED (0x04) // contiguous+escaped (0x05) or quoted+escaped (0x06)

typedef uint32_t state_t;

#define CHARACTER (0x00)
#define SPACE (0x01)
#define LINE_FEED (0x02)
#define DQUOTE (0x03)
#define SEMICOLON (0x04)
#define BACKSLASH (0x05)
#define BRACKET (0x06)

typedef struct {
  state_t current;
  uint32_t input;
  state_t next;
  uint32_t output;
} transition_t;

static const transition_t transitions[] = {
  // initial
  { INITIAL,             CHARACTER,   CONTIGUOUS,          1 },
  { INITIAL,             SPACE,       WHITESPACE,          1 },
  { INITIAL,             LINE_FEED,   INITIAL,             1 },
  { INITIAL,             DQUOTE,      QUOTED,              1 },
  { INITIAL,             SEMICOLON,   COMMENT,             0 },
  { INITIAL,             BACKSLASH,   CONTIGUOUS|ESCAPED,  1 },
  { INITIAL,             BRACKET,     INITIAL,             1 },
  // whitespace
  { WHITESPACE,          CHARACTER,   CONTIGUOUS,          1 },
  { WHITESPACE,          SPACE,       WHITESPACE,          0 },
  { WHITESPACE,          LINE_FEED,   INITIAL,             1 },
  { WHITESPACE,          DQUOTE,      QUOTED,              1 },
  { WHITESPACE,          SEMICOLON,   COMMENT,             0 },
  { WHITESPACE,          BACKSLASH,   CONTIGUOUS|ESCAPED,  1 },
  { WHITESPACE,          BRACKET,     WHITESPACE,          1 },
  // comment
  { COMMENT,             CHARACTER,   COMMENT,             0 },
  { COMMENT,             SPACE,       COMMENT,             0 },
  { COMMENT,             LINE_FEED,   INITIAL,             1 },
  { COMMENT,             DQUOTE,      COMMENT,             0 },
  { COMMENT,             SEMICOLON,   COMMENT,             0 },
  { COMMENT,             BACKSLASH,   COMMENT,             0 },
  { COMMENT,             BRACKET,     COMMENT,             0 },
  // contiguous
  { CONTIGUOUS,          CHARACTER,   CONTIGUOUS,          1 },
  { CONTIGUOUS,          SPACE,       WHITESPACE,          0 },
  { CONTIGUOUS,          LINE_FEED,   INITIAL,             1 },
  { CONTIGUOUS,          DQUOTE,      QUOTED,              1 },
  { CONTIGUOUS,          SEMICOLON,   COMMENT,             0 },
  { CONTIGUOUS,          BACKSLASH,   CONTIGUOUS|ESCAPED,  1 },
  { CONTIGUOUS,          BRACKET,     WHITESPACE,          1 },
  // escaped/contiguous
  { CONTIGUOUS|ESCAPED,  CHARACTER,   CONTIGUOUS,          1 },
  { CONTIGUOUS|ESCAPED,  SPACE,       CONTIGUOUS,          1 },
  { CONTIGUOUS|ESCAPED,  LINE_FEED,   CONTIGUOUS,          1 },
  { CONTIGUOUS|ESCAPED,  DQUOTE,      CONTIGUOUS,          1 },
  { CONTIGUOUS|ESCAPED,  SEMICOLON,   CONTIGUOUS,          1 },
  { CONTIGUOUS|ESCAPED,  BACKSLASH,   CONTIGUOUS,          1 },
  { CONTIGUOUS|ESCAPED,  BRACKET,     CONTIGUOUS,          1 },
  // quoted
  { QUOTED,              CHARACTER,   QUOTED,              1 },
  { QUOTED,              SPACE,       QUOTED,              1 },
  { QUOTED,              LINE_FEED,   QUOTED,              1 },
  { QUOTED,              DQUOTE,      WHITESPACE,          1 },
  { QUOTED,              SEMICOLON,   QUOTED,              1 },
  { QUOTED,              BACKSLASH,   QUOTED|ESCAPED,      1 },
  { QUOTED,              CHARACTER,   QUOTED,              1 },
  // escaped/quoted
  { QUOTED|ESCAPED,      CHARACTER,   QUOTED,              1 },
  { QUOTED|ESCAPED,      SPACE,       QUOTED,              1 },
  { QUOTED|ESCAPED,      LINE_FEED,   QUOTED,              1 },
  { QUOTED|ESCAPED,      DQUOTE,      QUOTED,              1 },
  { QUOTED|ESCAPED,      SEMICOLON,   QUOTED,              1 },
  { QUOTED|ESCAPED,      BACKSLASH,   QUOTED,              1 },
  { QUOTED|ESCAPED,      BRACKET,     QUOTED,              1 },
};

static int usage(const char *cmd)
{
  fprintf(stderr, "Usage: %s <output>\n", cmd);
  return 1;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
    return usage(argv[0]);

  // generate double packed states to cut the number of transitions

  int32_t max = 0;
  for (size_t i=0, n=sizeof(transitions)/sizeof(transitions[0]); i < n; i++) {
    if (transitions[i].current >= max)
      max = transitions[i].current + 1;
  }

  // allocate space for tables
  uint8_t **tables = calloc(max, sizeof(*tables));
  if (!tables)
    return 1;

  for (size_t i=0; i < max; i++) {
    if (!(tables[i] = calloc(UINT8_MAX+1, sizeof(**tables))))
      return 1;
  }

  for (size_t i=0, n=sizeof(transitions)/sizeof(transitions[0]); i < n; i++) {
    for (size_t j=0; j < n; j++) {
      if (transitions[i].next != transitions[j].current)
        continue;
      uint8_t index = (transitions[i].input << 4) | transitions[j].input;
      uint8_t output = (transitions[i].output << 1) | transitions[j].output;
      tables[transitions[i].current][index] =
        (output << 4) | (transitions[j].next & 0x0f);
    }
    // end-of-file
    uint8_t index = transitions[i].input << 4;
    uint8_t output = transitions[i].output;
    tables[transitions[i].current][index] =
      (output << 4) | (transitions[i].next & 0x0f);
  }

  FILE *output;

  if (!(output = fopen(argv[1], "wb")))
    return 1;

  fprintf(output,
    "#ifndef TRANSITIONS_H\n"
    "#define TRANSITIONS_H\n"
    "\n"
    "#include \"fsm.h\"\n"
    "\n"
    "static const transition_t transitions[%zu][%zu] = {\n",
    max, UINT8_MAX+1);

  for (size_t i=0; i < max; i++) {
    fprintf(output, "  {");
    for (size_t j=0, lf=0, n=UINT8_MAX+1; j < n; j++, lf++) {
      if (lf == 8)
        lf = 0, fprintf(output, "\n   ");
      uint8_t x = tables[i][j];
      fprintf(output, " {%u,%u}%s", x&0x0f, (x&0xf0)>>4, j+1 == n ? "" : ",");
    }
    fprintf(output, "\n  }%s\n", i+1 == max ? "" : ",");
  }

  fprintf(output,
    "};\n"
    "\n"
    "#endif // TRANSITIONS_H\n");

  fclose(output);

  return 0;
}
