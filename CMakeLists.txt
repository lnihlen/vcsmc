# Allow specifying project version.
cmake_policy(SET CMP0048 NEW)

project(vcsmc
        VERSION 0.0.1)

cmake_minimum_required(VERSION 3.9)

# To keep the version of clang/llvm used by Halide and vcsmc consistent we
# allow providing the same LLVM_CONFIG variable to this configuration.
if(DEFINED LLVM_CONFIG)
    message("using provided LLVM_CONFIG path ${LLVM_CONFIG}")
else()
    find_program(LLVM_CONFIG llvm-config)
endif()

execute_process(COMMAND "${LLVM_CONFIG}" --bindir
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE LLVM_CONFIG_RESULT
    OUTPUT_VARIABLE LLVM_BIN_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT LLVM_CONFIG_RESULT EQUAL "0")
    message(FATAL_ERROR "error running llvm-config at ${LLVM_CONFIG}, got result code ${LLVM_CONFIG_RESULT}")
endif()

message("detected ${LLVM_BIN_PATH} as binary path for llvm")

set(CMAKE_CXX_COMPILER "${LLVM_BIN_PATH}/clang++" CACHE FILEPATH "clang++" FORCE)
set(CMAKE_CXX_COMPILER_AR "${LLVM_BIN_PATH}/llvm-ar" CACHE FILEPATH "ar" FORCE)
set(CMAKE_CXX_COMPILER_RANLIB "${LLVM_BIN_PATH}/llvm-ranlib" CACHE FILEPATH "ranlib" FORCE)
set(CMAKE_C_COMPILER "${LLVM_BIN_PATH}/clang" CACHE FILEPATH "clang" FORCE)
set(CMAKE_C_COMPILER_AR "${LLVM_BIN_PATH}/llvm-ar" CACHE FILEPATH "ar" FORCE)
set(CMAKE_C_COMPILER_RANLIB "${LLVM_BIN_PATH}/llvm-ranlib" CACHE FILEPATH "ranlib" FORCE)
set(CMAKE_AR "${LLVM_BIN_PATH}/llvm-ar" CACHE FILEPATH "ar" FORCE)
set(CMAKE_LINKER "${LLVM_BIN_PATH}/llvm-ld" CACHE FILEPATH "ld" FORCE)
set(CMAKE_NM "{$LLVM_BIN_PATH}/llvm-nm" CACHE FILEPATH "nm" FORCE)
set(CMAKE_OBJDUMP "${LLVM_BIN_PATH}/llvm-objdump" CACHE FILEPATH "objdump" FORCE)
set(CMAKE_RANLIB "${LLVM_BIN_PATH}/llvm-ranlib" CACHE FILEPATH "ranlib" FORCE)

add_subdirectory(third_party)
add_subdirectory(src)

