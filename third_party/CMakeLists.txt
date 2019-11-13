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
# xxHash
set(XXHASH_BUNDLED_MODE ON CACHE BOOL "build xxhash as part of another project")
add_subdirectory(xxHash/cmake_unofficial)
target_include_directories(xxhash PUBLIC xxHash)

