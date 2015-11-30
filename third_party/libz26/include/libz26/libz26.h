#ifndef INCLUDE_LIBZ26_LIBZ26_H_
#define INCLUDE_LIBZ26_LIBZ26_H_

#include <stddef.h>  // For fixed-width types.
#include <stdint.h>  // For size_t.

// Call once per process, initializes the global tables that are (hopefully!)
// read-only.
void init_z26_global_tables(void);

const int kLibZ26ImageWidth = 320;
const int kLibZ26ImageHeight = 264;
const int kLibZ26ImageSizeBytes = kLibZ26ImageWidth * kLibZ26ImageHeight;

// Simulates a single frame based on the provided |byte_code|, then copies the
// output data into |output_picture|, which needs to be at least
// kLibZ26ImageSizeBytes large. TODO: define output format.
void simulate_single_frame(const uint8_t* byte_code,
                           size_t size,
                           uint8_t* output_picture);

#endif  // INCLUDE_LIBZ26_LIBZ26_H_
