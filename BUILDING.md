### Prerequisites

Building is currently only supported on Linux.
CMake is the build system.
Building Halide requires LLVM 8.0 or newer.

### Halide

This program uses [Halide](https://github.com/halide/Halide) as a toolchain for Ahead-of-Time compilation of vectorized
code for the image processing algorithms. Halide is a prerequisite for building vcsmc, so will need to be deployed
somewhere outside of the vcsmc source tree, and the HALIDE_DISTRIB_DIR variable needs to be supplied to cmake during
configuration.

Halide can be built from source, and a Halide *distribution* must be built. This form of Halide build currently does not
support cmake, but a ```make distrib``` command in the Halide source root directory will work. This should work, and
will produce a build of Halide in the ```distrib/``` subdirectory, which is where the cmake HALIDE_DISTRIB_DIR variable
should point.

### CMake Metabuild


