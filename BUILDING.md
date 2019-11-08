### Prerequisites

Building is currently only supported on Linux.
CMake is the build system.

### Halide

This program uses [Halide](https://github.com/halide/Halide) as a toolchain for Ahead-of-Time compilation of vectorized
code for the image processing algorithms. Halide is a prerequisite for building vcsmc, so will need to be deployed
somewhere outside of the vcsmc source tree, and the HALIDE_DISTRIB_DIR variable needs to be supplied to cmake during
configuration.

Halide can be built from source, and a Halide *distribution* must be built. This form of Halide build currently does not
support cmake, but a ```make distrib``` command in the Halide source root directory will work. This should work, and
will produce a build of Halide in the ```distrib/``` subdirectory, which is where the cmake HALIDE_DISTRIB_DIR variable
should point.

Building Halide requires LLVM 8.0 or newer. On Gentoo the system-installed version of llvm doesn't include some of the
source code files, and doesn't statically link many of the libraries that Halide requires. As a result it is easier to
download llvm and clang from source. Install clang in the tools/clang directory inside of the unpacked llvm source. It
feels like it would be prudent to match the version of llvm downloaded to the system-installed version. Then build llvm
and clang, according to the directions provided on the Halide README.md, and provide the path to the llvm-config tool
as the LLVM_CONFIG argument for the Halide make distrib command.

### CMake Metabuild

If using an LLVM build from source, you will also want to provide the same -DLLVM_CONFIG path to the cmake command line
as provided to Halide.

