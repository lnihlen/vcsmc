include("${HALIDE_DISTRIB_DIR}/halide.cmake")

add_subdirectory(halide)

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

add_library(vcsmc_common_lib
    ${vcsmc_common_src_files}
)

target_link_libraries(vcsmc_common_lib
    ${vcsmc_halide_libs}
    gflags::gflags
    xxhash
)

target_include_directories(vcsmc_common_lib PRIVATE
    "${HALIDE_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(epfg
    epfg.cc
    epfg_options.cc
    epfg_options.h
)

target_link_libraries(epfg PUBLIC
    vcsmc_common_lib
)

