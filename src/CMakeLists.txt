include("${HALIDE_DISTRIB_DIR}/halide.cmake")
# For the Halide library dependency.
link_directories("${HALIDE_DISTRIB_DIR}/lib")

# All the Halide stuff needs CPP11 to link properly.
set(CMAKE_CXX_STANDARD 11)

####
# Image Processing Library
add_library(vcsmc_image_processing STATIC
    atari_ntsc_rgb_color_table.cc
    atari_ntsc_rgb_color_table.h
    atari_ntsc_lab_color_table.h
    gaussian_kernel.cc
    gaussian_kernel.h
    VideoDecoder.cc
    VideoDecoder.h
)

set(vcsmc_halide_generators
    ciede_2k_generator.cc
    covariance_generator.cc
    mean_generator.cc
    rgb_to_lab_generator.cc
    ssim_generator.cc
    variance_generator.cc
)

target_include_directories(vcsmc_image_processing PUBLIC
    "${HALIDE_INCLUDE_DIR}"
)

foreach(GEN_SRC ${vcsmc_halide_generators})
    string(REPLACE "_generator.cc" "" GEN_NAME ${GEN_SRC})
    halide_library("${GEN_NAME}" SRCS ${GEN_SRC})
    target_link_libraries(vcsmc_image_processing PUBLIC "${GEN_NAME}")
endforeach()

####
# Generating atari_ntsc_lab_color_table depends on vcsmc_image_processing
add_executable(build_lab_table
    build_lab_table.cc
)

target_link_libraries(build_lab_table
    gflags
    Halide
    pthread
    vcsmc_image_processing
)

target_include_directories(build_lab_table PUBLIC
    "${HALIDE_INCLUDE_DIR}"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/atari_ntsc_lab_color_table.cc"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/build_lab_table" "--output_directory=${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS build_lab_table
)

####
# build_codon_table generates codon_table.{cc,h}
add_executable(build_codon_table
    build_codon_table.cc
)

target_link_libraries(build_codon_table
    gflags
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/codon_table.cc" "${CMAKE_CURRENT_BINARY_DIR}/codon_table.h"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/build_codon_table" "--output_directory=${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS build_codon_table
)

# Rest-of-project needs CPP17, particularly to use Pistache.
set(CMAKE_CXX_STANDARD 17)

####
# flatbuffer schema files
set(vcsmc_schema_files
)

build_flatbuffers(
    "${vcsmc_schema_files}"
    ""
    "vcsmc_schemas"
    ""
    "${CMAKE_CURRENT_BINARY_DIR}/schemas"
    ""
    ""
)

####
# vcsmc_core
set(vcsmc_core_files
    assembler.cc
    assembler.h
    bit_map.cc
    bit_map.h
    codon.h
    constants.h
    genome.cc
    genome.h
    gray_map.cc
    gray_map.h
    HttpEndpoint.cpp
    HttpEndpoint.h
    image_file.cc
    image_file.h
    kernel.h
    Logger.h
    Logger.cpp
    serialization.cc
    serialization.h
    snippet.h
    sound_file.cc
    sound_file.h
    sound.h
    spec.h
    state.cc
    state.h
    tls_prng.cc
    tls_prng.h
    types.h
    value_map.cc
    value_map.h
    Workflow.cpp
    Workflow.h
)

add_library(vcsmc_core STATIC
    ${vcsmc_core_files}
    "${CMAKE_CURRENT_BINARY_DIR}/codon_table.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/codon_table.h"
)

add_dependencies(vcsmc_core vcsmc_schemas)

target_link_libraries(vcsmc_core PUBLIC
    avcodec
    avformat
    avutil
    leveldb
    libz26
    pistache
    pistache_static
    png
    swscale
    xxhash
)

target_include_directories(vcsmc_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

####
# vcmsc_lib
add_library(vcsmc_lib STATIC
    "${CMAKE_CURRENT_BINARY_DIR}/atari_ntsc_lab_color_table.cc"
)

target_link_libraries(vcsmc_lib PUBLIC
    Halide
    vcsmc_core
    vcsmc_image_processing
)

####
# epfg
add_executable(epfg
    epfg.cc
)

target_link_libraries(epfg
    gflags::gflags
    vcsmc_lib
)

####
# run_unittests
set(vcsmc_unittests
    assembler_unittests.cc
    bit_map_unittests.cc
    ciede_2k_unittests.cc
    rgb_to_lab_unittests.cc
    serialization_unittests.cc
    ssim_unittests.cc
    state_unittests.cc
)

add_executable(run_unittests
    run_unittests.cc
    ${vcsmc_unittests}
)

target_link_libraries(run_unittests
    gtest
    vcsmc_lib
)

