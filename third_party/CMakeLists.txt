###
# benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "disable benchmark testing")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "disable install")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "don't depend on googletest")
add_subdirectory(benchmark)

###
# flatbuffers
add_subdirectory(flatbuffers)

###
# gflags
set(GFLAGS_BUILD_STATIC_LIBS ON CACHE BOOL "build gflags as a static library")
set(GFLAGS_BUILD_gflags_LIB ON CACHE BOOL "build multithreaded gflags library")
add_subdirectory(gflags)

###
# googletest
add_subdirectory(googletest)

###
# leveldb
set(LEVELDB_BUILD_TESTS OFF CACHE BOOL "build LevelDB's unit tests")
set(LEVELDB_BUILD_BENCHMARKS OFF CACHE BOOL "build LevelDB's benchmarks")
set(LEVELDB_INSTALL OFF CACHE BOOL "make LevelDB's install target")
add_subdirectory(leveldb)

###
# libz26
add_subdirectory(libz26)

###
#
set(PISTACHE_BUILD_EXAMPLES OFF CACHE BOOL "build Pistache examples")
set(PISTACHE_BUILD_TESTS OFF CACHE BOOL "build Pistache tests")
set(PISTACHE_ENABLE_NETWORK_TESTS OFF CACHE BOOL "enable Pistache network tests")
set(PISTACHE_USE_SSL OFF CACHE BOOL "enable ssl in Pistache")
add_subdirectory(pistache)

###
# xxHash
set(XXHASH_BUNDLED_MODE ON CACHE BOOL "build xxhash as part of another project")
add_subdirectory(xxHash/cmake_unofficial)
target_include_directories(xxhash PUBLIC xxHash)

