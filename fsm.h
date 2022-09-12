#ifndef FSM_H
#define FSM_H

typedef struct transition transition_t;
struct transition {
  uint8_t next, mask;
};

#endif // FSM_H
