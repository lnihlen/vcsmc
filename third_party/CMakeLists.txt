include(ExternalProject)

###
# googletest
add_subdirectory(googletest)

###
# gflags
set(GFLAGS_BUILD_STATIC_LIBS ON CACHE BOOL "build gflags as a static library")
set(GFLAGS_BUILD_gflags_LIB ON CACHE BOOL "build multithreaded gflags library")
add_subdirectory(gflags)

###
# Halide
# TODO: disable building extras
add_subdirectory(Halide)

# Variable to set to allow inclusion of halide.cmake for the halide_library
# cmake command.
set(HALIDE_DISTRIB_DIR "SETME")

###
# libz26
#patched version, build manually

###
# xxHash
set(XXHASH_BUNDLED_MODE ON CACHE BOOL "build xxhash as part of another project")
add_subdirectory(xxHash/cmake_unofficial)
target_include_directories(xxhash PUBLIC xxHash)


