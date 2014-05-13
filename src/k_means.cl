// Design ideas for openCL-ified k-means algorithm.

// General outline is pick k colors from Atari color set. (CPU)
// Calculate error distance from each color to each pixel (GPU)
// Each pixel gets class based on minimum error distance color. (either)
// The next step would be to calculate mean of each class but instead we
// calculate minimum distance color for each class - perhaps by just computing
// distances from all colors.

// CPU picks k colors at random, generates k 320-pixel wide color strips of
// those k colors, and uploads them to device, and runs them through rgb_to_lab.
// input line also gets uploaded (or is already part of image) and gets run
// through rbg_to_lab.

// Runs del_e distance for each pixel. (parallelize based on WorkGroup size?)

// Filter takes k inputs produces two outputs - minimum-error color, and that
// minimum error value. or could output array of [0, k) clases, or all 3.

// Then for each class find minimum-error color to approximate - compute error
// distance of that class to each color within the Atari pallette, choose min
// total error.

// Pick new k color set to represent the k classes, and repeat classification
// step, until no pixel changes class and no class color changes, or until
// some iteration limit is reached (in case solution is unstable).

// Might be good to run a couple of times with different random initial
// conditions to make sure we find best global minima.

