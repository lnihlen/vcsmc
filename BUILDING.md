### Prerequisites

Building is currently only supported on Linux.
CMake is the build system.
You will need to build llvm and clang from source.

We need the following ubuntu packages:
 * m4
 * libpng-dev
 * libz-dev
 * libavcodec-dev
 * libavformat-dev
 * libswscale-dev
 * libomp-dev
 * libelf-dev (for getting clang to build OpenMP support)
 * libffi-dev (same as above)

### Halide and LLVM

This program uses [Halide](https://github.com/halide/Halide) as a toolchain for Ahead-of-Time compilation of vectorized
code for the image processing algorithms. Halide is a prerequisite for building vcsmc, so will need to be deployed
somewhere outside of the vcsmc source tree, and the HALIDE_DISTRIB_DIR variable needs to be supplied to cmake during
configuration.

Halide can be built from source, and a Halide *distribution* must be built. This form of Halide build currently does not
support cmake, but a ```make distrib``` command in the Halide source root directory will work. This should work, and
will produce a build of Halide in the ```distrib/``` subdirectory, which is where the cmake HALIDE_DISTRIB_DIR variable
should point.

Building Halide requires LLVM 8.0 or newer. On some Linux systems the system-installed version of llvm doesn't include
some of the source code files, and may not statically link many of the libraries that Halide requires. As a result it
is easier to download llvm and clang from source. Build llvm and clang according to the directions provided on the Halide
README.md, with the additon of the "openmp" project to the -DLLVM_ENABLE_PROJECTS cmake flag. Once llvm is built, you
will want to install it somewhere with -DCMAKE_INSTALL_PREFIX and then a make install. Clang will not be able to find
the OpenMP support libraries if built in-tree. Provide the path to the *installed* llvm-config tool as the LLVM_CONFIG
argument for the Halide make distrib command.

CC=<path-to-installed-clang> CXX=<path-to-installed-clang++> LLVM_CONFIG=<path-to-installed-llvm-config>
    WITH_INTROSPECTION= make distrib

### CMake Metabuild

You will also want to provide the same -DLLVM_CONFIG path to the cmake command line as provided to Halide, along with
the -DHALIDE_DISTRIB_DIR from that Halide build.

cd vcsmc
mkdir build
cmake -DLLVM_CONFIG=<path> -DHALIDE_DISTRIB_DIR=<path> ..
make

