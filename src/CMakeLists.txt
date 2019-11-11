include("${HALIDE_DISTRIB_DIR}/halide.cmake")

add_executable(build_codon_table
    build_codon_table.cc
)

target_link_libraries(build_codon_table
    gflags
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/codon_table.h"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/build_codon_table" "--output_directory=${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS build_codon_table
)

link_directories("${HALIDE_DISTRIB_DIR}/lib")

set(vcsmc_common_src_files
    assembler.cc
    assembler.h
    bit_map.cc
    bit_map.h
    codon.h
    constants.h
    gaussian_kernel.cc
    gaussian_kernel.h
    genome.cc
    genome.h
    gray_map.cc
    gray_map.h
    image_file.cc
    image_file.h
    kernel.h
    serialization.cc
    serialization.h
    snippet.h
    sound_file.cc
    sound_file.h
    sound.h
    spec.h
    split.cc
    state.cc
    state.h
    tls_prng.cc
    tls_prng.h
    types.h
    value_map.cc
    value_map.h
    video_decoder.cc
    video_decoder.h
)

add_library(vcsmc_common_lib STATIC
    ${vcsmc_common_src_files}
    "${CMAKE_CURRENT_BINARY_DIR}/codon_table.h"
)

set(vcsmc_halide_generators
    ciede_2k_generator.cc
    covariance_generator.cc
    mean_generator.cc
    rgb_to_lab_generator.cc
    ssim_generator.cc
    variance_generator.cc
)

foreach(GEN_SRC ${vcsmc_halide_generators})
    string(REPLACE "_generator.cc" "" GEN_NAME ${GEN_SRC})
    halide_library("${GEN_NAME}" SRCS ${GEN_SRC})
    target_link_libraries(vcsmc_common_lib PUBLIC "${GEN_NAME}")
endforeach()

target_link_libraries(vcsmc_common_lib PUBLIC
    avcodec
    avformat
    avutil
    gflags::gflags
    Halide
    libz26
    png
    swscale
    xxhash
)

target_include_directories(vcsmc_common_lib PUBLIC
    "${HALIDE_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

add_executable(epfg
    epfg.cc
    epfg_options.cc
    epfg_options.h
)

target_link_libraries(epfg PUBLIC
    vcsmc_common_lib
)

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
    vcsmc_common_lib
    gtest
)

