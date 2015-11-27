#ifndef INCLUDE_LIBZ26_LIBZ26_H_
#define INCLUDE_LIBZ26_LIBZ26_H_

struct z26_state;

// Call once per process, initializes the global tables that are (hopefully!)
// read-only.
void init_z26_global_tables(void);

// Constructs a new simulator state structure, sets sane defaults, and returns
// it. Question - do we even need to bother exposing this?
struct z26_state* init_z26_state(void);

// Simulates a single frame.
void simulate_single_frame(struct z26_state* s);

#endif  // INCLUDE_LIBZ26_LIBZ26_H_
